//! Port of shap/cext/tree_shap.h: ensemble layout, prediction, path-dependent Tree SHAP
//! and path-dependent interaction values.

pub const TRANSFORM_IDENTITY: u32 = 0;
pub const TRANSFORM_LOGISTIC: u32 = 1;
pub const TRANSFORM_LOGISTIC_NLOGLOSS: u32 = 2;
pub const TRANSFORM_SQUARED_LOSS: u32 = 3;

pub type Transform = fn(f64, f64) -> f64;

fn logistic(margin: f64, _y: f64) -> f64 {
    1.0 / (1.0 + (-margin).exp())
}
fn logistic_nlogloss(margin: f64, y: f64) -> f64 {
    (1.0 + margin.exp()).ln() - y * margin
}
fn squared_loss(margin: f64, y: f64) -> f64 {
    (margin - y) * (margin - y)
}

pub fn get_transform(code: u32) -> Option<Transform> {
    match code {
        TRANSFORM_LOGISTIC => Some(logistic),
        TRANSFORM_LOGISTIC_NLOGLOSS => Some(logistic_nlogloss),
        TRANSFORM_SQUARED_LOSS => Some(squared_loss),
        _ => None,
    }
}

/// Flattened ensemble, same layout as shap's TreeEnsemble arrays: (n_trees, max_nodes[, n_outputs]).
pub struct Ensemble {
    pub cl: Vec<i32>,
    pub cr: Vec<i32>,
    pub cd: Vec<i32>,
    pub feat: Vec<i32>,
    pub thr: Vec<f64>,
    pub ttype: Vec<i32>,
    pub values: Vec<f64>,
    pub nsw: Vec<f64>,
    pub max_depth: usize,
    pub tree_limit: usize,
    pub base_offset: Vec<f64>,
    pub max_nodes: usize,
    pub nout: usize,
}

/// One tree of the ensemble (slices into the flat arrays).
#[derive(Clone, Copy)]
pub struct Tree<'a> {
    pub cl: &'a [i32],
    pub cr: &'a [i32],
    pub cd: &'a [i32],
    pub feat: &'a [i32],
    pub thr: &'a [f64],
    pub ttype: &'a [i32],
    pub values: &'a [f64],
    pub nsw: &'a [f64],
    pub nout: usize,
}

impl Ensemble {
    pub fn tree(&self, i: usize) -> Tree<'_> {
        let n = self.max_nodes;
        let d = i * n;
        Tree {
            cl: &self.cl[d..d + n],
            cr: &self.cr[d..d + n],
            cd: &self.cd[d..d + n],
            feat: &self.feat[d..d + n],
            thr: &self.thr[d..d + n],
            ttype: &self.ttype[d..d + n],
            values: &self.values[d * self.nout..(d + n) * self.nout],
            nsw: if self.nsw.is_empty() { &[] } else { &self.nsw[d..d + n] },
            nout: self.nout,
        }
    }
}

#[inline]
pub fn category_in_threshold(threshold: f64, category: f64) -> bool {
    let shift = (category as f32) as i32 - 1;
    let flag = 1i32.wrapping_shl(shift as u32);
    ((threshold as f32) as i32 & flag) != 0
}

impl<'a> Tree<'a> {
    #[inline(always)]
    pub fn next(&self, node: usize, x: &[f64], xm: &[bool]) -> usize {
        let f = self.feat[node] as usize;
        let tt = self.ttype[node];
        if tt == 0 {
            // numeric split, written as selects (no data-dependent branches)
            let left = x[f] <= self.thr[node];
            let n = if left { self.cl[node] } else { self.cr[node] };
            (if xm[f] { self.cd[node] } else { n }) as usize
        } else if xm[f] {
            self.cd[node] as usize
        } else if tt == 1 && category_in_threshold(self.thr[node], x[f]) {
            self.cl[node] as usize
        } else {
            self.cr[node] as usize
        }
    }

    /// Leaf index reached by x (tree_predict).
    #[inline]
    pub fn leaf(&self, x: &[f64], xm: &[bool]) -> usize {
        let mut node = 0usize;
        while self.cl[node] >= 0 {
            node = self.next(node, x, xm);
        }
        node
    }
}

/// dense_tree_predict for a single row: out[k] for k in 0..nout.
pub fn predict_row(e: &Ensemble, x: &[f64], xm: &[bool], y: f64, transform: Option<Transform>, out: &mut [f64]) {
    for k in 0..e.nout {
        out[k] += e.base_offset[k];
    }
    for j in 0..e.tree_limit {
        let t = e.tree(j);
        let leaf = t.leaf(x, xm);
        for k in 0..e.nout {
            out[k] += t.values[leaf * e.nout + k];
        }
    }
    if let Some(tf) = transform {
        for k in 0..e.nout {
            out[k] = tf(out[k], y);
        }
    }
}

/// compute_expectations for a single tree (values: n_nodes x nout, modified in place). Returns max depth.
pub fn compute_expectations(cl: &[i32], cr: &[i32], nsw: &[f64], values: &mut [f64], nout: usize, i: usize, depth: usize) -> usize {
    let mut max_depth = 0;
    if cr[i] >= 0 {
        let li = cl[i] as usize;
        let ri = cr[i] as usize;
        let dl = compute_expectations(cl, cr, nsw, values, nout, li, depth + 1);
        let dr = compute_expectations(cl, cr, nsw, values, nout, ri, depth + 1);
        let lw = nsw[li];
        let rw = nsw[ri];
        for j in 0..nout {
            if lw == 0.0 && rw == 0.0 {
                values[i * nout + j] = 0.0;
            } else {
                values[i * nout + j] = (lw * values[li * nout + j] + rw * values[ri * nout + j]) / (lw + rw);
            }
        }
        max_depth = dl.max(dr) + 1;
    }
    max_depth
}

// ---------------------------------------------------------------------------------------------
// Path-dependent Tree SHAP (Lundberg et al. Algorithm 2), faithful port of tree_shap_recursive.

const RECIP: [f64; 130] = {
    let mut a = [0.0f64; 130];
    let mut i = 1;
    while i < 130 {
        a[i] = 1.0 / i as f64;
        i += 1;
    }
    a
};

#[derive(Clone, Copy, Default)]
pub struct PathElement {
    fi: i32,
    z: f64,
    o: f64,
    w: f64,
}

#[inline]
fn extend_path(p: &mut [PathElement], ud: usize, z: f64, o: f64, fi: i32) {
    p[ud].fi = fi;
    p[ud].z = z;
    p[ud].o = o;
    p[ud].w = if ud == 0 { 1.0 } else { 0.0 };
    let d1 = (ud + 1) as f64;
    for i in (0..ud).rev() {
        p[i + 1].w += o * p[i].w * (i + 1) as f64 / d1;
        p[i].w = z * p[i].w * (ud - i) as f64 / d1;
    }
}

#[inline]
fn unwind_path(p: &mut [PathElement], ud: usize, pi: usize) {
    let o = p[pi].o;
    let z = p[pi].z;
    let mut n = p[ud].w;
    let d1 = (ud + 1) as f64;
    for i in (0..ud).rev() {
        if o != 0.0 {
            let tmp = p[i].w;
            p[i].w = n * d1 / ((i + 1) as f64 * o);
            n = tmp - p[i].w * z * (ud - i) as f64 / d1;
        } else {
            p[i].w = (p[i].w * d1) / (z * (ud - i) as f64);
        }
    }
    for i in pi..ud {
        p[i].fi = p[i + 1].fi;
        p[i].z = p[i + 1].z;
        p[i].o = p[i + 1].o;
    }
}

#[inline]
fn unwound_path_sum(p: &[PathElement], ud: usize, pi: usize) -> f64 {
    let o = p[pi].o;
    let z = p[pi].z;
    let mut n = p[ud].w;
    let mut total = 0.0;
    if o != 0.0 {
        for i in (0..ud).rev() {
            let tmp = n / ((i + 1) as f64 * o);
            total += tmp;
            n = p[i].w - tmp * z * (ud - i) as f64;
        }
    } else {
        for i in (0..ud).rev() {
            total += p[i].w / (z * (ud - i) as f64);
        }
    }
    total * (ud + 1) as f64
}

pub struct PathCtx<'a> {
    pub t: Tree<'a>,
    pub x: &'a [f64],
    pub xm: &'a [bool],
    pub cond: i32,
    pub cf: i32,
}

#[allow(clippy::too_many_arguments)]
fn tree_shap_recursive(
    c: &PathCtx,
    phi: &mut [f64],
    buf: &mut [PathElement],
    node: usize,
    ud_in: isize,
    parent_off: usize,
    pz: f64,
    po: f64,
    pfi: i32,
    cfrac: f64,
) {
    if cfrac == 0.0 {
        return;
    }
    let t = &c.t;
    let mut ud = ud_in;
    let off = parent_off + (ud + 1) as usize;
    buf.copy_within(parent_off..parent_off + (ud + 1) as usize, off);
    if c.cond == 0 || c.cf != pfi {
        extend_path(&mut buf[off..], ud as usize, pz, po, pfi);
    }
    let split = t.feat[node];

    if t.cr[node] < 0 {
        let nout = t.nout;
        let vo = node * nout;
        let p = &buf[off..];
        let udu = ud as usize;
        // Same quantity as unwound_path_sum(p, udu, i) for every i, restructured: for elements with
        // one_fraction == 0 the sum sum_k w_k / (z (udu-k)) shares sum_k w_k/(udu-k) across elements,
        // and divisions by small integers become multiplications by reciprocals.
        let r = &RECIP;
        let d1 = (udu + 1) as f64;
        let mut s0 = f64::NAN;
        for i in 1..=udu {
            let el = p[i];
            let w = if udu >= 129 {
                unwound_path_sum(p, udu, i)
            } else if el.o != 0.0 {
                let inv_o = 1.0 / el.o;
                let mut n = p[udu].w;
                let mut total = 0.0;
                for k in (0..udu).rev() {
                    let tmp = n * r[k + 1] * inv_o;
                    total += tmp;
                    n = p[k].w - tmp * el.z * (udu - k) as f64;
                }
                total * d1
            } else {
                if s0.is_nan() {
                    s0 = 0.0;
                    for k in 0..udu {
                        s0 += p[k].w * r[udu - k];
                    }
                }
                s0 / el.z * d1
            };
            let po_ = el.fi as usize * nout;
            let scale = w * (el.o - el.z) * cfrac;
            for j in 0..nout {
                phi[po_ + j] += scale * t.values[vo + j];
            }
        }
    } else {
        let hot = t.next(node, c.x, c.xm);
        let cold = if hot as i32 == t.cl[node] { t.cr[node] as usize } else { t.cl[node] as usize };
        let w = t.nsw[node];
        let hz = t.nsw[hot] / w;
        let cz = t.nsw[cold] / w;
        let mut iz = 1.0;
        let mut io = 1.0;

        let udu = ud as usize;
        let mut pi = 0usize;
        while pi <= udu {
            if buf[off + pi].fi as u32 == split as u32 {
                break;
            }
            pi += 1;
        }
        if pi != udu + 1 {
            iz = buf[off + pi].z;
            io = buf[off + pi].o;
            unwind_path(&mut buf[off..], udu, pi);
            ud -= 1;
        }

        let mut hcf = cfrac;
        let mut ccf = cfrac;
        if c.cond > 0 && split == c.cf {
            ccf = 0.0;
            ud -= 1;
        } else if c.cond < 0 && split == c.cf {
            hcf *= hz;
            ccf *= cz;
            ud -= 1;
        }

        tree_shap_recursive(c, phi, buf, hot, ud + 1, off, hz * iz, io, split, hcf);
        tree_shap_recursive(c, phi, buf, cold, ud + 1, off, cz * iz, 0.0, split, ccf);
    }
}

pub fn path_buf(max_depth: usize) -> Vec<PathElement> {
    let maxd = max_depth + 2;
    vec![PathElement::default(); (maxd * (maxd + 1)) / 2 + 1]
}

/// tree_shap: adds one tree's contribution into phi ((M+1) x nout).
pub fn tree_shap(t: Tree, x: &[f64], xm: &[bool], m: usize, phi: &mut [f64], cond: i32, cf: i32, buf: &mut [PathElement]) {
    if cond == 0 {
        for j in 0..t.nout {
            phi[m * t.nout + j] += t.values[j];
        }
    }
    let c = PathCtx { t, x, xm, cond, cf };
    tree_shap_recursive(&c, phi, buf, 0, 0, 0, 1.0, 1.0, -1, 1.0);
}

/// Row-level path-dependent SHAP: out has (M+1)*nout entries, zero-initialised.
pub fn path_dependent_row(e: &Ensemble, x: &[f64], xm: &[bool], m: usize, out: &mut [f64], buf: &mut [PathElement]) {
    for j in 0..e.tree_limit {
        tree_shap(e.tree(j), x, xm, m, out, 0, 0, buf);
    }
    for j in 0..e.nout {
        out[m * e.nout + j] += e.base_offset[j];
    }
}

/// Unique features per tree, exactly as dense_tree_interactions_path_dependent builds them.
pub fn unique_features(e: &Ensemble, m: usize) -> (Vec<i32>, usize) {
    let amount = m.min(e.max_nodes);
    let mut uf = vec![-1i32; e.tree_limit * amount];
    for j in 0..e.tree_limit {
        let fr = &e.feat[j * e.max_nodes..(j + 1) * e.max_nodes];
        let ur = &mut uf[j * amount..(j + 1) * amount];
        for k in 0..e.max_nodes {
            for l in 0..amount {
                if fr[k] == ur[l] {
                    break;
                }
                if ur[l] < 0 {
                    ur[l] = fr[k];
                    break;
                }
            }
        }
    }
    (uf, amount)
}

/// Row-level interaction values: out has (M+1)*(M+1)*nout entries, zero-initialised.
#[allow(clippy::too_many_arguments)]
pub fn interactions_row(
    e: &Ensemble,
    uf: &[i32],
    amount: usize,
    x: &[f64],
    xm: &[bool],
    m: usize,
    out: &mut [f64],
    buf: &mut [PathElement],
    diag: &mut [f64],
    on: &mut [f64],
    off: &mut [f64],
) {
    let crs = (m + 1) * e.nout;
    diag.fill(0.0);
    for j in 0..e.tree_limit {
        let t = e.tree(j);
        tree_shap(t, x, xm, m, diag, 0, 0, buf);
        let ur = &uf[j * amount..(j + 1) * amount];
        for &ind in ur {
            if ind < 0 {
                break;
            }
            on.fill(0.0);
            off.fill(0.0);
            tree_shap(t, x, xm, m, on, 1, ind, buf);
            tree_shap(t, x, xm, m, off, -1, ind, buf);
            let base = ind as usize * crs;
            for l in 0..crs {
                let val = (on[l] - off[l]) / 2.0;
                out[base + l] += val;
                diag[l] -= val;
            }
        }
    }
    for j in 0..m + 1 {
        let o = j * crs + j * e.nout;
        for k in 0..e.nout {
            out[o + k] = diag[j * e.nout + k];
        }
    }
    let last = (m * (m + 1) + m) * e.nout;
    for j in 0..e.nout {
        out[last + j] += e.base_offset[j];
    }
}

/// tree_saabas / dense_tree_saabas for one row (out: (M+1) x nout).
pub fn saabas_row(e: &Ensemble, x: &[f64], xm: &[bool], m: usize, out: &mut [f64]) {
    let nout = e.nout;
    for j in 0..e.tree_limit {
        let t = e.tree(j);
        let mut cur = 0usize;
        while t.cl[cur] >= 0 {
            let f = t.feat[cur] as usize;
            let next = if xm[f] {
                t.cd[cur] as usize
            } else if x[f] <= t.thr[cur] {
                t.cl[cur] as usize
            } else {
                t.cr[cur] as usize
            };
            for i in 0..nout {
                out[f * nout + i] += t.values[next * nout + i] - t.values[cur * nout + i];
            }
            cur = next;
        }
    }
    for j in 0..nout {
        out[m * nout + j] += e.base_offset[j];
    }
}

pub fn extend_path_pub(p: &mut [PathElement], ud: usize, z: f64, o: f64, fi: i32) {
    extend_path(p, ud, z, o, fi)
}

pub fn unwound_path_sum_pub(p: &[PathElement], ud: usize, pi: usize) -> f64 {
    unwound_path_sum(p, ud, pi)
}

#[inline]
pub fn path_w(p: &[PathElement], i: usize) -> f64 {
    p[i].w
}
