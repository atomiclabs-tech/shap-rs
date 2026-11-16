//! Fast interventional Tree SHAP for the identity transform.
//!
//! tree_shap_indep explores the "hybrid" paths of x and a reference r. Unrolled, its result is a sum
//! over leaves: leaf l (value v) is reachable iff every feature j on the leaf's path is satisfied by x
//! (bit in X) or by r (bit in R). With S_x = X & !R, S_r = R & !X, M = |S_x|, N = |S_x| + |S_r|:
//!   features in S_x get  v * w(N, M-1),  features in S_r get -v * w(N, M)
//! (the float32 products shap computes; w = memoized weights), and the bias gets v when M == 0,
//! which is exactly r's own leaf. So for one tree, the sum over all references only depends on the
//! per-leaf histogram of R masks: per leaf we build a table indexed by X (or keep the distinct R
//! masks when 2^d is too large), and each row costs one pass over the nodes plus lookups.
//! The per-leaf terms are bit-identical to shap's; only the order of summation differs.

use crate::indep::{directions, Node};
use crate::tree::Ensemble;
use std::collections::HashMap;

enum Kind {
    Table(usize),            // offset into tables: tab[off + X*d + i]
    List(usize, usize),      // range into rlist
}

pub(crate) struct Leaf {
    pub(crate) node: usize,
    pub(crate) feats: Vec<u32>,
    kind: Kind,
}

pub(crate) struct TreeFast {
    pub(crate) bit: Vec<u8>,
    pub(crate) internal: Vec<u32>,
    pub(crate) leaves: Vec<Leaf>,
    tab: Vec<f64>,
    rlist: Vec<(u32, f64)>,
    pub(crate) value: Vec<f32>, // leaf values by leaf index
}

/// Fail masks for one row: fail[leaf] has bit j set if the row violates a split on local feature j.
pub(crate) fn fail_pass(tf: &TreeFast, dirs: &[u8], tree: &[Node], fail: &mut [u32]) {
    fail[0] = 0;
    for &node in &tf.internal {
        let nu = node as usize;
        let nd = tree[nu];
        let (hot, cold) = match dirs[nu] {
            0 => (nd.cl, nd.cr),
            1 => (nd.cr, nd.cl),
            2 => (nd.cd, if nd.cd == nd.cl { nd.cr } else { nd.cl }),
            _ => (-1, -1),
        };
        let f = fail[nu];
        let b = 1u32 << tf.bit[nu];
        if hot < 0 {
            fail[nd.cl as usize] = f | b;
            fail[nd.cr as usize] = f | b;
        } else {
            fail[hot as usize] = f;
            fail[cold as usize] = f | b;
        }
    }
}

const TABLE_MAX_D: usize = 12;

/// Leaf list (unique path features as local bits), internal nodes in DFS preorder, local bit per node.
/// None if a path has more than 32 distinct features. Stumps have no leaves.
pub(crate) fn structure(tree: &[Node]) -> Option<TreeFast> {
    let n = tree.len();
    let mut bit = vec![0u8; n];
    let mut internal = Vec::new();
    let mut leaves = Vec::new();
    let mut value = Vec::new();
    if tree[0].cl < 0 {
        return Some(TreeFast { bit, internal, leaves, tab: vec![], rlist: vec![], value });
    }
    let mut stack: Vec<(usize, Vec<u32>)> = vec![(0, vec![])];
    while let Some((node, feats)) = stack.pop() {
        let nd = tree[node];
        if nd.cl < 0 {
            if feats.len() > 31 {
                return None;
            }
            leaves.push(Leaf { node, feats, kind: Kind::List(0, 0) });
            value.push(nd.value);
            continue;
        }
        internal.push(node as u32);
        let f = nd.feat as u32;
        let pos = feats.iter().position(|&g| g == f);
        let b = pos.unwrap_or(feats.len());
        if b > 31 {
            return None;
        }
        bit[node] = b as u8;
        let mut fe = feats;
        if pos.is_none() {
            fe.push(f);
        }
        stack.push((nd.cl as usize, fe.clone()));
        stack.push((nd.cr as usize, fe));
    }
    Some(TreeFast { bit, internal, leaves, tab: vec![], rlist: vec![], value })
}

fn build(tree: &[Node], rdirs: &[Vec<u8>], mw: &[f32], md: usize, n_rows: usize) -> Option<(TreeFast, f64)> {
    let n = tree.len();
    if tree[0].cl < 0 {
        // stump: only the bias
        let v = tree[0].value as f64 * rdirs.len() as f64;
        return Some((structure(tree)?, v));
    }
    let mut tf = structure(tree)?;

    // histogram of reference masks per leaf, and the bias (sum of each reference's own leaf value)
    let mut fail = vec![0u32; n];
    let mut hist: Vec<HashMap<u32, f64>> = vec![HashMap::new(); tf.leaves.len()];
    let mut bias = 0.0f64;
    for rd in rdirs {
        fail_pass(&tf, rd, tree, &mut fail);
        for (li, l) in tf.leaves.iter().enumerate() {
            let full = if l.feats.len() == 32 { u32::MAX } else { (1u32 << l.feats.len()) - 1 };
            let r = full & !fail[l.node];
            if r == full {
                bias += l_value(&tf, li) as f64;
            }
            *hist[li].entry(r).or_insert(0.0) += 1.0;
        }
    }

    let w = |nn: u32, mm: u32| -> f32 { mw[(nn + md as u32 * mm) as usize] };
    for li in 0..tf.leaves.len() {
        let d = tf.leaves[li].feats.len();
        if d == 0 {
            continue;
        }
        let v = tf.value[li];
        let full = if d == 32 { u32::MAX } else { (1u32 << d) - 1 };
        let mut rs: Vec<(u32, f64)> = hist[li].iter().map(|(&k, &c)| (k, c)).collect();
        rs.sort_by_key(|e| e.0);
        if d <= TABLE_MAX_D && (1usize << d) <= n_rows.max(4) {
            let off = tf.tab.len();
            tf.tab.resize(off + (1usize << d) * d, 0.0);
            let tab = &mut tf.tab[off..];
            for x in 0..=full {
                for &(r, cnt) in &rs {
                    if (x | r) != full {
                        continue;
                    }
                    let sx = x & !r;
                    let sr = r & !x;
                    let mm = sx.count_ones();
                    let nn = mm + sr.count_ones();
                    if nn == 0 {
                        continue;
                    }
                    let pos = if mm != 0 { v * w(nn, mm - 1) } else { 0.0 };
                    let neg = if mm != nn { -v * w(nn, mm) } else { 0.0 };
                    for i in 0..d {
                        let b = 1u32 << i;
                        if sx & b != 0 {
                            tab[x as usize * d + i] += cnt * pos as f64;
                        } else if sr & b != 0 {
                            tab[x as usize * d + i] += cnt * neg as f64;
                        }
                    }
                }
            }
            tf.leaves[li].kind = Kind::Table(off);
        } else {
            let start = tf.rlist.len();
            tf.rlist.extend(rs);
            tf.leaves[li].kind = Kind::List(start, tf.rlist.len() - start);
        }
    }
    Some((tf, bias))
}

fn l_value(tf: &TreeFast, li: usize) -> f32 {
    tf.value[li]
}

fn row(tf: &TreeFast, dirs: &[u8], tree: &[Node], fail: &mut [u32], mw: &[f32], md: usize, out: &mut [f64]) {
    if tf.leaves.is_empty() {
        return;
    }
    fail_pass(tf, dirs, tree, fail);
    for (li, l) in tf.leaves.iter().enumerate() {
        let d = l.feats.len();
        if d == 0 {
            continue;
        }
        let full = if d == 32 { u32::MAX } else { (1u32 << d) - 1 };
        let x = full & !fail[l.node];
        match l.kind {
            Kind::Table(off) => {
                let t = &tf.tab[off + x as usize * d..off + (x as usize + 1) * d];
                for i in 0..d {
                    out[l.feats[i] as usize] += t[i];
                }
            }
            Kind::List(start, len) => {
                let v = tf.value[li];
                for &(r, cnt) in &tf.rlist[start..start + len] {
                    if (x | r) != full {
                        continue;
                    }
                    let sx = x & !r;
                    let sr = r & !x;
                    let mm = sx.count_ones();
                    let nn = mm + sr.count_ones();
                    if nn == 0 {
                        continue;
                    }
                    let pos = if mm != 0 { (v * mw[(nn + md as u32 * (mm - 1)) as usize]) as f64 } else { 0.0 };
                    let neg = if mm != nn { (-v * mw[(nn + md as u32 * mm) as usize]) as f64 } else { 0.0 };
                    for i in 0..d {
                        let b = 1u32 << i;
                        if sx & b != 0 {
                            out[l.feats[i] as usize] += cnt * pos;
                        } else if sr & b != 0 {
                            out[l.feats[i] as usize] += cnt * neg;
                        }
                    }
                }
            }
        }
    }
}

/// Interventional SHAP (identity transform) for output `oind`, written into out[i*row + k*nout + oind].
/// Returns false if some tree can't use the fast path (caller falls back to the exact port).
#[allow(clippy::too_many_arguments)]
pub fn independent_identity(
    e: &Ensemble,
    nodes: &[Node],
    mw: &[f32],
    oind: usize,
    xs: &[f64],
    xms: &[bool],
    rs: &[f64],
    rms: &[bool],
    n: usize,
    m: usize,
    num_r: usize,
    out: &mut [f64],
) -> bool {
    use rayon::prelude::*;
    let mn = e.max_nodes;
    let md = e.max_depth;
    let nout = e.nout;
    let row_len = (m + 1) * nout;
    let res: Option<(Vec<f64>, f64)> = (0..e.tree_limit)
        .into_par_iter()
        .fold(
            || Some((vec![0.0f64; n * (m + 1)], 0.0f64)),
            |acc, k| {
                let (mut acc, mut bias) = acc?;
                let tree = &nodes[k * mn..(k + 1) * mn];
                let rdirs: Vec<Vec<u8>> = (0..num_r)
                    .map(|j| {
                        let mut d = vec![3u8; mn];
                        directions(tree, &rs[j * m..(j + 1) * m], &rms[j * m..(j + 1) * m], &mut d);
                        d
                    })
                    .collect();
                let (tf, b) = build(tree, &rdirs, mw, md, n)?;
                bias += b;
                let mut dirs = vec![3u8; mn];
                let mut fail = vec![0u32; mn];
                for i in 0..n {
                    directions(tree, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], &mut dirs);
                    row(&tf, &dirs, tree, &mut fail, mw, md, &mut acc[i * (m + 1)..(i + 1) * (m + 1)]);
                }
                Some((acc, bias))
            },
        )
        .reduce(
            || Some((vec![0.0f64; n * (m + 1)], 0.0f64)),
            |a, b| {
                let (mut a, ba) = a?;
                let (b, bb) = b?;
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                Some((a, ba + bb))
            },
        );
    let Some((acc, bias)) = res else { return false };
    let inv = num_r as f64;
    for i in 0..n {
        for k in 0..m {
            out[i * row_len + k * nout + oind] = acc[i * (m + 1) + k] / inv;
        }
        out[i * row_len + m * nout + oind] = e.base_offset[oind] + bias / inv;
    }
    true
}
