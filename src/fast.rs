//! Fast TreeSHAP v2-style path-dependent SHAP (Yang 2021, "Fast TreeSHAP").
//!
//! For path-dependent Tree SHAP the contribution of leaf l to feature i is
//!     v_l * (o_i - z_i) * W_i(A \ {i})
//! where z_j is the product of cover fractions of feature j's splits on the path, o_j in {0,1} says
//! whether x follows all of feature j's splits on the path, A = {j : o_j = 1}, and W_i is the
//! unwound permutation weight. W only depends on the leaf and A, so per tree we precompute a table
//! over all 2^d subsets of each leaf's d unique features (with the same extend/unwind arithmetic
//! as the original algorithm); explaining a row is then a DFS computing A per leaf plus a lookup.

use crate::tree::{path_w, extend_path_pub, path_buf, tree_shap, unwound_path_sum_pub, Ensemble, PathElement, Tree};

pub struct LeafTable {
    pub node: usize,
    pub feats: Vec<u32>, // global feature index of local bit j
    pub off: usize,      // offset into weights: weights[off + A*d + i]
    pub zs: Vec<f64>,
    pub doff: usize, // offset into `done` (lazy tables): done[doff + A]
}

pub struct TreeTable {
    pub bit: Vec<u8>, // per internal node: local bit index of its split feature on the root path
    pub leaves: Vec<LeafTable>,
    pub leaf_of_node: Vec<u32>, // node -> index into leaves (u32::MAX if internal)
    pub weights: Vec<f64>,
    pub internal: Vec<u32>, // internal nodes, parents before children
    pub premul: bool,       // weights already multiplied by the leaf value (single output)
    pub done: Vec<bool>,    // lazy tables: entry (leaf, A) computed
    pub inodes: Vec<INode>, // internal nodes in `internal` order, packed (numeric-only trees)
}

#[derive(Clone, Copy)]
pub struct INode {
    pub thr: f64,
    pub feat: u32,
    pub idx: u32,
    pub cl: u32,
    pub cr: u32,
    pub cd: u32,
    pub mask: u32,
}

/// Packed internal nodes when every split is numeric (else empty: use Tree::next).
pub fn pack(t: &Tree, internal: &mut [u32], bit: &[u8]) -> Vec<INode> {
    // breadth-first order: parents still precede children, but consecutive nodes are independent
    let mut depth = vec![0u32; t.cl.len()];
    for &n in internal.iter() {
        let n = n as usize;
        depth[t.cl[n] as usize] = depth[n] + 1;
        depth[t.cr[n] as usize] = depth[n] + 1;
    }
    internal.sort_by_key(|&n| depth[n as usize]);
    if internal.iter().any(|&n| t.ttype[n as usize] != 0) {
        return vec![];
    }
    internal
        .iter()
        .map(|&n| {
            let n = n as usize;
            INode {
                thr: t.thr[n],
                feat: t.feat[n] as u32,
                idx: n as u32,
                cl: t.cl[n] as u32,
                cr: t.cr[n] as u32,
                cd: t.cd[n] as u32,
                mask: 1u32 << bit[n],
            }
        })
        .collect()
}

/// fail[node] for every node: bit j set if x violates a split on local feature j above the node.
#[inline]
pub fn fail_pass(t: &Tree, internal: &[u32], bit: &[u8], inodes: &[INode], x: &[f64], xm: &[bool], fail: &mut [u32]) {
    fail[0] = 0;
    if !inodes.is_empty() || internal.is_empty() {
        for nd in inodes {
            let f = nd.feat as usize;
            let left = x[f] <= nd.thr;
            let (mut hot, mut cold) = if left { (nd.cl, nd.cr) } else { (nd.cr, nd.cl) };
            if xm[f] {
                hot = nd.cd;
                cold = if nd.cd == nd.cl { nd.cr } else { nd.cl };
            }
            let fv = fail[nd.idx as usize];
            fail[hot as usize] = fv;
            fail[cold as usize] = fv | nd.mask;
            fail[nd.idx as usize] = fv | if hot == nd.cl { HOT_LEFT } else { 0 };
        }
        return;
    }
    for &node in internal {
        let node = node as usize;
        let hot = t.next(node, x, xm);
        let cold = if hot as i32 == t.cl[node] { t.cr[node] as usize } else { t.cl[node] as usize };
        let f = fail[node];
        fail[hot] = f;
        fail[cold] = f | (1u32 << bit[node]);
        fail[node] = f | if hot as i32 == t.cl[node] { HOT_LEFT } else { 0 };
    }
}

/// Collect per-leaf unique features and zero fractions; returns None if the table would be too big.
pub fn build(t: &Tree, max_entries: usize, lazy_rows: Option<usize>) -> Option<TreeTable> {
    let n = t.cl.len();
    let mut bit = vec![0u8; n];
    let mut leaf_of_node = vec![u32::MAX; n];
    let mut leaves = Vec::new();
    let mut zs_all: Vec<Vec<f64>> = Vec::new();
    // iterative DFS: (node, feats, zs)
    let mut stack: Vec<(usize, Vec<u32>, Vec<f64>)> = vec![(0, vec![], vec![])];
    let mut total = 0usize;
    let mut internal: Vec<u32> = Vec::new();
    while let Some((node, feats, zs)) = stack.pop() {
        if t.cr[node] < 0 {
            let d = feats.len();
            if d > 24 {
                return None;
            }
            total += (1usize << d) * d.max(1);
            if total > max_entries {
                return None;
            }
            leaf_of_node[node] = leaves.len() as u32;
            leaves.push(LeafTable { node, feats, off: 0, zs: zs.clone(), doff: 0 });
            zs_all.push(zs);
            continue;
        }
        internal.push(node as u32);
        let f = t.feat[node] as u32;
        let w = t.nsw[node];
        let pos = feats.iter().position(|&g| g == f);
        let b = pos.unwrap_or(feats.len());
        if b > 255 {
            return None;
        }
        bit[node] = b as u8;
        for child in [t.cl[node] as usize, t.cr[node] as usize] {
            let mut fe = feats.clone();
            let mut z = zs.clone();
            let frac = t.nsw[child] / w;
            match pos {
                Some(p) => z[p] *= frac,
                None => {
                    fe.push(f);
                    z.push(frac);
                }
            }
            stack.push((child, fe, z));
        }
    }
    let mut off = 0;
    let mut doff = 0;
    for l in leaves.iter_mut() {
        l.off = off;
        l.doff = doff;
        off += (1usize << l.feats.len()) * l.feats.len();
        doff += 1usize << l.feats.len();
    }
    // lazy mode builds its own (paired) tables in tree_rows_lazy
    let mut weights = if lazy_rows.is_some() { Vec::new() } else { vec![0.0f64; off] };
    let premul = t.nout == 1;
    // lazy filling pays off when rows can only touch a fraction of the 2^d entries per leaf
    let lazy = lazy_rows.is_some() && std::env::var("SHAP_RS_EAGER").is_err();
    if lazy {
        let inodes = pack(t, &mut internal, &bit);
        return Some(TreeTable { bit, leaves, leaf_of_node, weights, internal, premul, done: vec![], inodes });
    }
    let maxd = leaves.iter().map(|l| l.feats.len()).max().unwrap_or(0);
    // one path buffer per depth level so prefixes are shared across subsets
    let mut bufs: Vec<Vec<PathElement>> = vec![vec![PathElement::default(); maxd + 2]; maxd + 2];
    for (li, l) in leaves.iter().enumerate() {
        let d = l.feats.len();
        if d == 0 {
            continue;
        }
        let zs = &zs_all[li];
        let v = if premul { t.values[l.node] } else { 1.0 };
        let tab = &mut weights[l.off..l.off + (1usize << d) * d];
        extend_path_pub(&mut bufs[0], 0, 1.0, 1.0, -1);
        fill(&mut bufs, zs, d, 0, 0, tab, v);
    }
    let inodes = pack(t, &mut internal, &bit);
    Some(TreeTable { bit, leaves, leaf_of_node, weights, internal, premul, done: vec![true; doff], inodes })
}

/// Depth-first over the bits of A: level j decides o_j; extensions are shared by all subsets
/// with the same prefix.
fn fill(bufs: &mut [Vec<PathElement>], zs: &[f64], d: usize, j: usize, a: usize, tab: &mut [f64], v: f64) {
    if j == d {
        let p = &bufs[d];
        for i in 0..d {
            let o = ((a >> i) & 1) as f64;
            let w = unwound_path_sum_pub(p, d, i + 1);
            tab[a * d + i] = w * (o - zs[i]) * v;
        }
        return;
    }
    for o in [0usize, 1] {
        let (lo, hi) = bufs.split_at_mut(j + 1);
        hi[0][..j + 1].copy_from_slice(&lo[j][..j + 1]);
        extend_path_pub(&mut hi[0], j + 1, zs[j], o as f64, j as i32);
        fill(bufs, zs, d, j + 1, a | (o << j), tab, v);
    }
}

/// Add one tree's path-dependent SHAP values for row x into phi ((M+1) x nout), bias excluded.
pub fn row(t: &Tree, tab: &TreeTable, x: &[f64], xm: &[bool], phi: &mut [f64], fail: &mut Vec<u32>) {
    let nout = t.nout;
    if fail.len() < t.cl.len() {
        fail.resize(t.cl.len(), 0);
    }
    fail[0] = 0;
    for &node in &tab.internal {
        let node = node as usize;
        let hot = t.next(node, x, xm);
        let cold = if hot as i32 == t.cl[node] { t.cr[node] as usize } else { t.cl[node] as usize };
        let f = fail[node];
        fail[hot] = f;
        fail[cold] = f | (1u32 << tab.bit[node]);
    }
    for l in &tab.leaves {
        let d = l.feats.len();
        if d == 0 {
            continue;
        }
        let a = ((1u32 << d) - 1) & !fail[l.node];
        let w = &tab.weights[l.off + a as usize * d..l.off + (a as usize + 1) * d];
        if tab.premul {
            for i in 0..d {
                phi[l.feats[i] as usize] += w[i];
            }
        } else {
            let vo = l.node * nout;
            for i in 0..d {
                let fo = l.feats[i] as usize * nout;
                for k in 0..nout {
                    phi[fo + k] += w[i] * t.values[vo + k];
                }
            }
        }
    }
}

/// Estimated work of the original algorithm vs the table approach for one tree and n rows.
pub fn worth_it(t: &Tree, n_rows: usize) -> bool {
    // depth/unique-feature profile via DFS
    let mut stack: Vec<(usize, Vec<u32>)> = vec![(0, vec![])];
    let (mut v1, mut pre) = (0f64, 0f64);
    let mut nodes = 0f64;
    while let Some((node, feats)) = stack.pop() {
        nodes += 1.0;
        if t.cr[node] < 0 {
            let d = feats.len() as f64;
            v1 += (d + 1.0) * (d + 1.0) + 2.0 * d;
            pre += 2f64.powf(d) * (d + 1.0) * (d + 1.0) * 1.5;
            if feats.len() > 20 {
                return false;
            }
            continue;
        }
        v1 += 3.0 * feats.len() as f64 + 4.0;
        let f = t.feat[node] as u32;
        let mut fe = feats.clone();
        if !fe.contains(&f) {
            fe.push(f);
        }
        stack.push((t.cl[node] as usize, fe.clone()));
        stack.push((t.cr[node] as usize, fe));
    }
    let n = n_rows as f64;
    let v2 = pre + n * nodes * 1.5;
    v2 < n * v1
}

/// Full path-dependent SHAP (same output as path_dependent_row for every row) using per-tree tables
/// where that is cheaper, and the original recursion otherwise.
pub fn path_dependent(e: &Ensemble, xs: &[f64], xms: &[bool], n: usize, m: usize) -> Vec<f64> {
    use rayon::prelude::*;
    let nout = e.nout;
    let row = (m + 1) * nout;
    let max_entries = 1usize << 23; // 64 MB of f64 per tree table
    let threads = rayon::current_num_threads();

    let process_tree = |k: usize, acc: &mut [f64], buf: &mut Vec<PathElement>, _stack: &mut Vec<u32>| {
        let t = e.tree(k);
        match build(&t, max_entries, Some(n)) {
            Some(mut tb) => {
                for i in 0..n {
                    for j in 0..nout {
                        acc[i * row + m * nout + j] += t.values[j];
                    }
                }
                tree_rows_lazy(&t, &mut tb, xs, xms, n, m, acc, buf);
            }
            None => {
                for i in 0..n {
                    let o = &mut acc[i * row..(i + 1) * row];
                    tree_shap(t, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], m, o, 0, 0, buf);
                }
            }
        }
    };

    let mut out = if e.tree_limit >= 2 * threads || n < 2 * threads {
        (0..e.tree_limit)
            .into_par_iter()
            .fold(
                || (vec![0.0f64; n * row], path_buf(e.max_depth), Vec::new()),
                |(mut acc, mut buf, mut st), k| {
                    process_tree(k, &mut acc, &mut buf, &mut st);
                    (acc, buf, st)
                },
            )
            .map(|(acc, _, _)| acc)
            .reduce(|| vec![0.0f64; n * row], |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            })
    } else {
        // few trees, many rows: parallelise rows within each tree
        let mut out = vec![0.0f64; n * row];
        for k in 0..e.tree_limit {
            let t = e.tree(k);
            let tab = if worth_it(&t, n) { build(&t, max_entries, None) } else { None };
            out.par_chunks_mut(row).enumerate().for_each_init(
                || (path_buf(e.max_depth), Vec::new()),
                |(buf, st), (i, o)| {
                    let x = &xs[i * m..(i + 1) * m];
                    let xm = &xms[i * m..(i + 1) * m];
                    match &tab {
                        Some(tb) => {
                            for j in 0..nout {
                                o[m * nout + j] += t.values[j];
                            }
                            row_fn(&t, tb, x, xm, o, st);
                        }
                        None => tree_shap(t, x, xm, m, o, 0, 0, buf),
                    }
                },
            );
        }
        out
    };
    for i in 0..n {
        for j in 0..nout {
            out[i * row + m * nout + j] += e.base_offset[j];
        }
    }
    out
}

/// Lookup unit: a single leaf, or a pair of sibling leaves under `node` sharing one table indexed by
/// (parent mask, hot direction). Entries hold weight * leaf value for every (feature, output).
struct Unit {
    node: usize,
    l: usize,
    r: usize, // usize::MAX for a single leaf
    d: usize,
    off: usize,
    doff: usize,
}

const HOT_LEFT: u32 = 1 << 31;

/// All rows of one tree with lazily filled tables: masks for a block of rows first, then
/// unit-outer / row-inner so each table stays in cache; entries are computed on first use.
#[allow(clippy::too_many_arguments)]
fn tree_rows_lazy(t: &Tree, tb: &mut TreeTable, xs: &[f64], xms: &[bool], n: usize, m: usize, acc: &mut [f64], buf: &mut Vec<PathElement>) {
    let nout = t.nout;
    let row = (m + 1) * nout;
    // parents of leaves, to pair siblings
    let mut parent = vec![u32::MAX; t.cl.len()];
    for &p in &tb.internal {
        parent[t.cl[p as usize] as usize] = p;
        parent[t.cr[p as usize] as usize] = p;
    }
    let mut units: Vec<Unit> = Vec::new();
    let mut paired = vec![false; tb.leaves.len()];
    let (mut off, mut doff) = (0usize, 0usize);
    for li in 0..tb.leaves.len() {
        let d = tb.leaves[li].feats.len();
        if d == 0 || paired[li] {
            continue;
        }
        let node = tb.leaves[li].node;
        let p = parent[node];
        let mut unit = Unit { node, l: li, r: usize::MAX, d, off, doff };
        if p != u32::MAX {
            let (cl, cr) = (t.cl[p as usize] as usize, t.cr[p as usize] as usize);
            let (lo, ro) = (tb.leaf_of_node[cl], tb.leaf_of_node[cr]);
            if lo != u32::MAX && ro != u32::MAX && d < 24 {
                unit = Unit { node: p as usize, l: lo as usize, r: ro as usize, d, off, doff };
                paired[lo as usize] = true;
                paired[ro as usize] = true;
            }
        }
        let slots = if unit.r == usize::MAX { 1usize << d } else { 2usize << d };
        off += slots * d * nout;
        doff += slots;
        units.push(unit);
    }
    let mut weights = vec![0.0f64; off];
    let mut done = vec![false; doff];
    let mut tmp = vec![0.0f64; 32];

    let nu = units.len();
    const BLOCK: usize = 64;
    let mut fail = vec![0u32; t.cl.len()];
    let mut fails = vec![0u32; BLOCK * nu];
    if buf.len() < 40 {
        buf.resize(40, PathElement::default());
    }
    for b0 in (0..n).step_by(BLOCK) {
        let b1 = (b0 + BLOCK).min(n);
        for i in b0..b1 {
            let x = &xs[i * m..(i + 1) * m];
            let xm = &xms[i * m..(i + 1) * m];
            fail_pass(t, &tb.internal, &tb.bit, &tb.inodes, x, xm, &mut fail);
            for (ui, u) in units.iter().enumerate() {
                fails[ui * BLOCK + (i - b0)] = fail[u.node];
            }
        }
        for (ui, u) in units.iter().enumerate() {
            let d = u.d;
            let full = (1u32 << d) - 1;
            let feats = &tb.leaves[u.l].feats;
            let slot_len = d * nout;
            for i in b0..b1 {
                let f = fails[ui * BLOCK + (i - b0)];
                let slot = if u.r == usize::MAX {
                    (full & !f) as usize
                } else {
                    (((full & !(f & !HOT_LEFT)) as usize) << 1) | (f >> 31) as usize
                };
                let w0 = u.off + slot * slot_len;
                if !done[u.doff + slot] {
                    done[u.doff + slot] = true;
                    let w = &mut weights[w0..w0 + slot_len];
                    if u.r == usize::MAX {
                        let l = &tb.leaves[u.l];
                        entry(&l.zs, d, slot, 1.0, &mut tmp[..d], buf);
                        for q in 0..d {
                            for k in 0..nout {
                                w[q * nout + k] = tmp[q] * t.values[l.node * nout + k];
                            }
                        }
                    } else {
                        let ap = (slot >> 1) as u32;
                        let hl = slot & 1 == 1;
                        let bp = 1u32 << tb.bit[u.node];
                        for (li, a) in [(u.l, if hl { ap } else { ap & !bp }), (u.r, if hl { ap & !bp } else { ap })] {
                            let l = &tb.leaves[li];
                            entry(&l.zs, d, a as usize, 1.0, &mut tmp[..d], buf);
                            for q in 0..d {
                                for k in 0..nout {
                                    w[q * nout + k] += tmp[q] * t.values[l.node * nout + k];
                                }
                            }
                        }
                    }
                }
                let w = &weights[w0..w0 + slot_len];
                let o = &mut acc[i * row..(i + 1) * row];
                if nout == 1 {
                    for q in 0..d {
                        o[feats[q] as usize] += w[q];
                    }
                } else {
                    for q in 0..d {
                        let fo = feats[q] as usize * nout;
                        for k in 0..nout {
                            o[fo + k] += w[q * nout + k];
                        }
                    }
                }
            }
        }
    }
}

/// One table entry: weights of all d features for satisfied-set a (times v). Same arithmetic as
/// unwound_path_sum, with the o == 0 sum shared across features.
fn entry(zs: &[f64], d: usize, a: usize, v: f64, out: &mut [f64], buf: &mut [PathElement]) {
    extend_path_pub(buf, 0, 1.0, 1.0, -1);
    for j in 0..d {
        extend_path_pub(buf, j + 1, zs[j], ((a >> j) & 1) as f64, j as i32);
    }
    let d1 = (d + 1) as f64;
    let mut s0 = f64::NAN;
    for i in 0..d {
        let o = ((a >> i) & 1) as f64;
        let w = if o != 0.0 {
            unwound_path_sum_pub(buf, d, i + 1)
        } else {
            if s0.is_nan() {
                s0 = 0.0;
                for k in 0..d {
                    s0 += path_w(buf, k) / (d - k) as f64;
                }
            }
            s0 / zs[i] * d1
        };
        out[i] = w * (o - zs[i]) * v;
    }
}

#[inline]
fn row_fn(t: &Tree, tab: &TreeTable, x: &[f64], xm: &[bool], phi: &mut [f64], stack: &mut Vec<u32>) {
    row(t, tab, x, xm, phi, stack)
}

#[allow(dead_code)]
fn _unused(_: &Tree) {}
#[allow(dead_code)]
fn _u2(_: Tree) {}
