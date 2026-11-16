//! Fast interventional Tree SHAP with an output transform (model_output="probability" / "log_loss").
//!
//! shap's dense_independent sums tree_shap_indep over all trees for one (x, r) pair, then scales the
//! pair's vector by s(x, r) = (T(m_x) - T(m_r)) / (m_x - m_r) (1 when the margins are equal). The per-pair
//! scaling prevents the background histogram trick used for the identity transform, but the sum is
//! still linear in the per-leaf terms, so phi_x = (1/R) sum_trees sum_r s(x, r) * leaf terms(x, r).
//! With the unrolled leaf formula (see fast_indep.rs), leaf l contributes for reference r iff every
//! path feature is satisfied by x or by r. Per (tree, x) a DFS follows x while carrying the list of
//! references that satisfy every split x fails ("alive"); only nodes with a non-empty alive list are
//! visited, and at a leaf each alive reference costs a few bit operations plus one update per feature
//! where x and r differ. The bias is sum_r T(base + B_r, 0) with B_r = r's own leaf values, independent
//! of x. Leaf terms are the same float32 products as shap's; only the summation order differs.

use crate::fast_indep::{fail_pass, structure, TreeFast};
use crate::indep::{directions, Node};
use crate::tree::{Ensemble, Transform};
use rayon::prelude::*;

#[inline(always)]
fn pick(d: u8, cl: i32, cr: i32, cd: i32) -> i32 {
    match d {
        0 => cl,
        1 => cr,
        2 => cd,
        _ => -1,
    }
}

struct Ctx<'a> {
    tree: &'a [Node],
    st: &'a TreeFast,
    leaf_of: &'a [u32],
    rchild: &'a [i32],
    rmask: &'a [u32],
    nl: usize,
    mn: usize,
    xdir: &'a [u8],
    xfail: &'a [u32],
    srow: &'a [f64],
    mw: &'a [f32],
    md: u32,
}

fn rec(c: &Ctx, node: usize, arena: &mut Vec<u32>, start: usize, len: usize, acc: &mut [f64]) {
    let nd = c.tree[node];
    if nd.cl < 0 {
        let li = c.leaf_of[node] as usize;
        let l = &c.st.leaves[li];
        let d = l.feats.len();
        if d == 0 {
            return;
        }
        let full = if d == 32 { u32::MAX } else { (1u32 << d) - 1 };
        let x = full & !c.xfail[node];
        let v = c.st.value[li];
        for &r in &arena[start..start + len] {
            let rm = c.rmask[r as usize * c.nl + li];
            if (x | rm) != full {
                continue;
            }
            let sx = x & !rm;
            let sr = rm & !x;
            let mm = sx.count_ones();
            let nn = mm + sr.count_ones();
            if nn == 0 {
                continue;
            }
            let sc = c.srow[r as usize];
            if mm != 0 {
                let pos = (v * c.mw[(nn + c.md * (mm - 1)) as usize]) as f64 * sc;
                let mut b = sx;
                while b != 0 {
                    let i = b.trailing_zeros() as usize;
                    acc[l.feats[i] as usize] += pos;
                    b &= b - 1;
                }
            }
            if mm != nn {
                let neg = (-v * c.mw[(nn + c.md * mm) as usize]) as f64 * sc;
                let mut b = sr;
                while b != 0 {
                    let i = b.trailing_zeros() as usize;
                    acc[l.feats[i] as usize] += neg;
                    b &= b - 1;
                }
            }
        }
        return;
    }
    let hot = pick(c.xdir[node], nd.cl, nd.cr, nd.cd);
    for child in [nd.cl, nd.cr] {
        if child == hot {
            rec(c, child as usize, arena, start, len, acc);
        } else {
            let s2 = arena.len();
            for k in start..start + len {
                let r = arena[k];
                if c.rchild[r as usize * c.mn + node] == child {
                    arena.push(r);
                }
            }
            let l2 = arena.len() - s2;
            if l2 > 0 {
                rec(c, child as usize, arena, s2, l2, acc);
            }
            arena.truncate(s2);
        }
    }
}

/// One tree, rows b0..b1: adds s(x, r)-scaled leaf terms into a ((b1-b0) x (m+1)).
#[allow(clippy::too_many_arguments)]
fn tree_rows(
    tree: &[Node],
    st: &TreeFast,
    mw: &[f32],
    md: u32,
    xs: &[f64],
    xms: &[bool],
    rs: &[f64],
    rms: &[bool],
    b0: usize,
    b1: usize,
    m: usize,
    num_r: usize,
    scale: &[f64],
    a: &mut [f64],
) {
    let mn = tree.len();
    let nl = st.leaves.len();
    let mut leaf_of = vec![u32::MAX; mn];
    for (li, l) in st.leaves.iter().enumerate() {
        leaf_of[l.node] = li as u32;
    }
    // per reference: child taken at every node, and the satisfied-feature mask at every leaf
    let mut rchild = vec![-1i32; num_r * mn];
    let mut rmask = vec![0u32; num_r * nl];
    let mut dir = vec![3u8; mn];
    let mut fail = vec![0u32; mn];
    for j in 0..num_r {
        directions(tree, &rs[j * m..(j + 1) * m], &rms[j * m..(j + 1) * m], &mut dir);
        let rc = &mut rchild[j * mn..(j + 1) * mn];
        for (node, nd) in tree.iter().enumerate() {
            if nd.cl >= 0 {
                rc[node] = pick(dir[node], nd.cl, nd.cr, nd.cd);
            }
        }
        fail_pass(st, &dir, tree, &mut fail);
        for (li, l) in st.leaves.iter().enumerate() {
            let d = l.feats.len();
            let full = if d == 32 { u32::MAX } else { (1u32 << d) - 1 };
            rmask[j * nl + li] = full & !fail[l.node];
        }
    }
    let mut arena: Vec<u32> = Vec::with_capacity(num_r * (tree.len().min(64) + 2));
    for i in b0..b1 {
        directions(tree, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], &mut dir);
        fail_pass(st, &dir, tree, &mut fail);
        arena.clear();
        arena.extend(0..num_r as u32);
        let c = Ctx {
            tree,
            st,
            leaf_of: &leaf_of,
            rchild: &rchild,
            rmask: &rmask,
            nl,
            mn,
            xdir: &dir,
            xfail: &fail,
            srow: &scale[i * num_r..(i + 1) * num_r],
            mw,
            md,
        };
        rec(&c, 0, &mut arena, 0, num_r, &mut a[(i - b0) * (m + 1)..(i - b0 + 1) * (m + 1)]);
    }
}

/// Interventional SHAP with a transform for output `oind`, written into out[i*row + k*nout + oind].
/// Returns false (nothing written) if some tree can't use the fast path.
#[allow(clippy::too_many_arguments)]
pub fn independent_transform(
    e: &Ensemble,
    nodes: &[Node],
    mw: &[f32],
    oind: usize,
    xs: &[f64],
    xms: &[bool],
    ys: Option<&[f64]>,
    rs: &[f64],
    rms: &[bool],
    n: usize,
    m: usize,
    num_r: usize,
    margins_x: &[f64],
    margins_r: &[f64],
    tf: Transform,
    out: &mut [f64],
) -> bool {
    let mn = e.max_nodes;
    let md = e.max_depth as u32;
    let structs: Option<Vec<TreeFast>> = (0..e.tree_limit).into_par_iter().map(|k| structure(&nodes[k * mn..(k + 1) * mn])).collect();
    let Some(structs) = structs else { return false };

    // per-pair rescale factors, exactly as dense_independent computes them
    let mut scale = vec![1.0f64; n * num_r];
    scale.par_chunks_mut(num_r.max(1)).enumerate().for_each(|(i, srow)| {
        let mx = margins_x[i];
        let yi = ys.map(|v| v[i]).unwrap_or(0.0);
        let tx = tf(mx, yi);
        for (j, s) in srow.iter_mut().enumerate().take(num_r) {
            let mr = margins_r[j];
            *s = if mx == mr { 1.0 } else { (tx - tf(mr, yi)) / (mx - mr) };
        }
    });

    // bias: B_r = sum over trees (in order) of r's own leaf value (float32 node values)
    let bias_terms: Vec<f64> = (0..num_r)
        .into_par_iter()
        .map(|j| {
            let r = &rs[j * m..(j + 1) * m];
            let rm = &rms[j * m..(j + 1) * m];
            let mut b = 0.0f64;
            let mut dir = vec![3u8; mn];
            for k in 0..e.tree_limit {
                let tree = &nodes[k * mn..(k + 1) * mn];
                directions(tree, r, rm, &mut dir);
                let mut node = 0i32;
                loop {
                    let nd = tree[node as usize];
                    if nd.cl < 0 {
                        b += nd.value as f64;
                        break;
                    }
                    let c = pick(dir[node as usize], nd.cl, nd.cr, nd.cd);
                    if c < 0 {
                        break;
                    }
                    node = c;
                }
            }
            tf(e.base_offset[oind] + b, 0.0)
        })
        .collect();
    let mut bias = 0.0f64;
    for b in &bias_terms {
        bias += b;
    }

    // trees in parallel (thread-local accumulators), rows in blocks to bound accumulator memory
    let block = (1usize << 21) / (m + 1);
    let mut acc = vec![0.0f64; n * (m + 1)];
    for b0 in (0..n).step_by(block.max(1)) {
        let b1 = (b0 + block.max(1)).min(n);
        let nb = b1 - b0;
        let part = structs
            .par_iter()
            .enumerate()
            .filter(|(_, st)| !st.leaves.is_empty())
            .fold(
                || vec![0.0f64; nb * (m + 1)],
                |mut a, (k, st)| {
                    tree_rows(&nodes[k * mn..(k + 1) * mn], st, mw, md, xs, xms, rs, rms, b0, b1, m, num_r, &scale, &mut a);
                    a
                },
            )
            .reduce(
                || vec![0.0f64; nb * (m + 1)],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += y;
                    }
                    a
                },
            );
        acc[b0 * (m + 1)..b1 * (m + 1)].copy_from_slice(&part);
    }

    let nout = e.nout;
    let row_len = (m + 1) * nout;
    let inv = num_r as f64;
    for i in 0..n {
        for k in 0..m {
            out[i * row_len + k * nout + oind] = acc[i * (m + 1) + k] / inv;
        }
        out[i * row_len + m * nout + oind] = bias / inv;
    }
    true
}
