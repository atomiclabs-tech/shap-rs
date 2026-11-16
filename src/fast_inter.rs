//! Fast path-dependent SHAP interaction values.
//!
//! dense_tree_interactions_path_dependent runs, per tree and per unique feature u, two conditioned
//! Tree SHAP passes (u "on": cold branches of u get weight 0; u "off": both branches weighted by the
//! cover fraction) and takes (on - off) / 2. Per leaf with unique features F, zero fractions z and
//! one-indicators o (x follows all of that feature's splits), this is
//!     phi[u][i] = v * (o_u - z_u) * (o_i - z_i) * W_i(path without u) / 2
//! which is symmetric in (u, i). So per row and leaf we build the leaf's path once, unwind each u
//! once and read every pair from it: O(d^3 / 2) per leaf instead of 2 * |U| full tree traversals.
//! The diagonal is (SHAP value) - (row sum of off-diagonal terms), as in shap.

use crate::tree::{extend_path_pub, unwound_path_sum_pub, Ensemble, PathElement, Tree};

pub struct LeafInfo {
    node: usize,
    feats: Vec<u32>,
    zs: Vec<f64>,
}

pub struct TreeInfo {
    bit: Vec<u8>,
    internal: Vec<u32>,
    leaves: Vec<LeafInfo>,
}

pub fn build(t: &Tree) -> Option<TreeInfo> {
    let n = t.cl.len();
    let mut bit = vec![0u8; n];
    let mut internal = Vec::new();
    let mut leaves = Vec::new();
    let mut stack: Vec<(usize, Vec<u32>, Vec<f64>)> = vec![(0, vec![], vec![])];
    while let Some((node, feats, zs)) = stack.pop() {
        if t.cr[node] < 0 {
            if feats.len() > 31 {
                return None;
            }
            leaves.push(LeafInfo { node, feats, zs });
            continue;
        }
        internal.push(node as u32);
        let f = t.feat[node] as u32;
        let w = t.nsw[node];
        let pos = feats.iter().position(|&g| g == f);
        let b = pos.unwrap_or(feats.len());
        if b > 31 {
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
    Some(TreeInfo { bit, internal, leaves })
}

pub struct Scratch {
    fail: Vec<u32>,
    full: Vec<PathElement>,
    un: Vec<PathElement>,
}

impl Scratch {
    pub fn new(max_nodes: usize, max_depth: usize) -> Self {
        Scratch {
            fail: vec![0; max_nodes],
            full: vec![PathElement::default(); max_depth + 3],
            un: vec![PathElement::default(); max_depth + 3],
        }
    }
}

/// Interaction values of one row: out ((M+1) x (M+1) x nout), zero-initialised; `shap` scratch ((M+1) x nout).
pub fn row(e: &Ensemble, infos: &[TreeInfo], x: &[f64], xm: &[bool], m: usize, out: &mut [f64], shap: &mut [f64], s: &mut Scratch) {
    let nout = e.nout;
    let crs = (m + 1) * nout;
    shap.fill(0.0);
    for (k, info) in infos.iter().enumerate() {
        let t = e.tree(k);
        for j in 0..nout {
            shap[m * nout + j] += t.values[j];
        }
        let fail = &mut s.fail;
        fail[0] = 0;
        for &node in &info.internal {
            let node = node as usize;
            let hot = t.next(node, x, xm);
            let cold = if hot as i32 == t.cl[node] { t.cr[node] as usize } else { t.cl[node] as usize };
            let f = fail[node];
            fail[hot] = f;
            fail[cold] = f | (1u32 << info.bit[node]);
        }
        for l in &info.leaves {
            let d = l.feats.len();
            if d == 0 {
                continue;
            }
            let fl = fail[l.node];
            let o = |j: usize| if fl & (1u32 << j) == 0 { 1.0 } else { 0.0 };
            let vo = l.node * nout;
            // full path: root + d unique features
            extend_path_pub(&mut s.full, 0, 1.0, 1.0, -1);
            for j in 0..d {
                extend_path_pub(&mut s.full, j + 1, l.zs[j], o(j), j as i32);
            }
            // main effects (plain Tree SHAP)
            for i in 0..d {
                let w = unwound_path_sum_pub(&s.full, d, i + 1) * (o(i) - l.zs[i]);
                let fo = l.feats[i] as usize * nout;
                for kk in 0..nout {
                    shap[fo + kk] += w * t.values[vo + kk];
                }
            }
            // pairs
            for u in 0..d {
                let cu = o(u) - l.zs[u];
                if u + 1 >= d {
                    break;
                }
                // path without u (what the conditioned recursion builds)
                extend_path_pub(&mut s.un, 0, 1.0, 1.0, -1);
                let mut ud = 0;
                for j in 0..d {
                    if j != u {
                        ud += 1;
                        extend_path_pub(&mut s.un, ud, l.zs[j], o(j), j as i32);
                    }
                }
                let fu = l.feats[u] as usize;
                for i in u + 1..d {
                    // feature i (> u) sits at path index i once u is left out
                    let w = unwound_path_sum_pub(&s.un, d - 1, i) * (o(i) - l.zs[i]) * cu / 2.0;
                    let fi = l.feats[i] as usize;
                    for kk in 0..nout {
                        let val = w * t.values[vo + kk];
                        out[fu * crs + fi * nout + kk] += val;
                        out[fi * crs + fu * nout + kk] += val;
                    }
                }
            }
        }
    }
    // diagonal = SHAP value minus the off-diagonal row sum
    for j in 0..m {
        for kk in 0..nout {
            let mut off = 0.0;
            for u in 0..m {
                if u != j {
                    off += out[u * crs + j * nout + kk];
                }
            }
            out[j * crs + j * nout + kk] = shap[j * nout + kk] - off;
        }
    }
    for kk in 0..nout {
        out[m * crs + m * nout + kk] = shap[m * nout + kk] + e.base_offset[kk];
    }
}

/// Per-leaf weights for a given satisfied-set mask: d main-effect weights followed by the
/// d(d-1)/2 pair weights (u < i), all without the leaf value.
fn leaf_weights(l: &LeafInfo, fl: u32, s: &mut Scratch, w: &mut Vec<f64>) {
    let d = l.feats.len();
    let o = |j: usize| if fl & (1u32 << j) == 0 { 1.0 } else { 0.0 };
    w.clear();
    extend_path_pub(&mut s.full, 0, 1.0, 1.0, -1);
    for j in 0..d {
        extend_path_pub(&mut s.full, j + 1, l.zs[j], o(j), j as i32);
    }
    for i in 0..d {
        w.push(unwound_path_sum_pub(&s.full, d, i + 1) * (o(i) - l.zs[i]));
    }
    for u in 0..d.saturating_sub(1) {
        let cu = o(u) - l.zs[u];
        extend_path_pub(&mut s.un, 0, 1.0, 1.0, -1);
        let mut ud = 0;
        for j in 0..d {
            if j != u {
                ud += 1;
                extend_path_pub(&mut s.un, ud, l.zs[j], o(j), j as i32);
            }
        }
        for i in u + 1..d {
            w.push(unwound_path_sum_pub(&s.un, d - 1, i) * (o(i) - l.zs[i]) * cu / 2.0);
        }
    }
}

/// Tree-parallel variant with memoisation of leaf weights per distinct mask across rows.
pub fn interactions_memo(e: &Ensemble, infos: &[TreeInfo], xs: &[f64], xms: &[bool], n: usize, m: usize) -> Vec<f64> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let nout = e.nout;
    let crs = (m + 1) * nout;
    let row_len = (m + 1) * crs;
    let (mut out, shap) = (0..infos.len())
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n * row_len], vec![0.0f64; n * crs]),
            |(mut out, mut shap), k| {
                let info = &infos[k];
                let t = e.tree(k);
                let mut s = Scratch::new(e.max_nodes, e.max_depth.max(32));
                let mut caches: Vec<HashMap<u32, Vec<f64>>> = vec![HashMap::new(); info.leaves.len()];
                let mut wbuf = Vec::new();
                for i in 0..n {
                    let x = &xs[i * m..(i + 1) * m];
                    let xm = &xms[i * m..(i + 1) * m];
                    let o = &mut out[i * row_len..(i + 1) * row_len];
                    let sh = &mut shap[i * crs..(i + 1) * crs];
                    for j in 0..nout {
                        sh[m * nout + j] += t.values[j];
                    }
                    let fail = &mut s.fail;
                    fail[0] = 0;
                    for &node in &info.internal {
                        let node = node as usize;
                        let hot = t.next(node, x, xm);
                        let cold = if hot as i32 == t.cl[node] { t.cr[node] as usize } else { t.cl[node] as usize };
                        let f = fail[node];
                        fail[hot] = f;
                        fail[cold] = f | (1u32 << info.bit[node]);
                    }
                    for (li, l) in info.leaves.iter().enumerate() {
                        let d = l.feats.len();
                        if d == 0 {
                            continue;
                        }
                        let fl = s.fail[l.node];
                        if !caches[li].contains_key(&fl) {
                            leaf_weights(l, fl, &mut s, &mut wbuf);
                            caches[li].insert(fl, wbuf.clone());
                        }
                        let w = &caches[li][&fl];
                        let vo = l.node * nout;
                        for a in 0..d {
                            let fo = l.feats[a] as usize * nout;
                            for kk in 0..nout {
                                sh[fo + kk] += w[a] * t.values[vo + kk];
                            }
                        }
                        let mut p = d;
                        for u in 0..d {
                            let fu = l.feats[u] as usize;
                            for b in u + 1..d {
                                let fi = l.feats[b] as usize;
                                for kk in 0..nout {
                                    let val = w[p] * t.values[vo + kk];
                                    o[fu * crs + fi * nout + kk] += val;
                                    o[fi * crs + fu * nout + kk] += val;
                                }
                                p += 1;
                            }
                        }
                    }
                }
                (out, shap)
            },
        )
        .reduce(
            || (vec![0.0f64; n * row_len], vec![0.0f64; n * crs]),
            |(mut a, mut sa), (b, sb)| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                sa.iter_mut().zip(&sb).for_each(|(x, y)| *x += y);
                (a, sa)
            },
        );
    for i in 0..n {
        let o = &mut out[i * row_len..(i + 1) * row_len];
        let sh = &shap[i * crs..(i + 1) * crs];
        for j in 0..m {
            for kk in 0..nout {
                let mut off = 0.0;
                for u in 0..m {
                    if u != j {
                        off += o[u * crs + j * nout + kk];
                    }
                }
                o[j * crs + j * nout + kk] = sh[j * nout + kk] - off;
            }
        }
        for kk in 0..nout {
            o[m * crs + m * nout + kk] = sh[m * nout + kk] + e.base_offset[kk];
        }
    }
    out
}
