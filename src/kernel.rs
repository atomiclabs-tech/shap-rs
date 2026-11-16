//! KernelSHAP: port of shap/explainers/_kernel.py (coalition enumeration/sampling, kernel weights,
//! expected values, LARS feature selection (sklearn's lars_path, method="lar") and the constrained
//! weighted least squares solve). The user's model is called from Python between plan and solve.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::lasso;

// ------------------------------------------------------------------------------------------------
// numpy's legacy global RNG (np.random.RandomState / MT19937). shap's KernelExplainer draws its
// sampled coalitions with np.random.choice(p=...) and np.random.permutation; replaying the same
// MT19937 stream from np.random.get_state() reproduces shap's draws exactly.
pub struct Mt {
    pub key: Vec<u32>,
    pub pos: usize,
}
impl Mt {
    fn gen(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        let k = &mut self.key;
        let mix = |a: u32, b: u32, c: u32| -> u32 {
            let y = (a & 0x8000_0000) | (b & 0x7fff_ffff);
            c ^ (y >> 1) ^ ((y & 1).wrapping_neg() & 0x9908_b0df)
        };
        for i in 0..N - M {
            k[i] = mix(k[i], k[i + 1], k[i + M]);
        }
        for i in N - M..N - 1 {
            k[i] = mix(k[i], k[i + 1], k[i + M - N]);
        }
        k[N - 1] = mix(k[N - 1], k[0], k[M - 1]);
        self.pos = 0;
    }
    fn next_u32(&mut self) -> u32 {
        if self.pos >= 624 {
            self.gen();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    /// random_sample()
    fn next_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }
    /// legacy random_interval(max): masked rejection sampling in [0, max]
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }
    /// permutation(n) (arange + legacy Fisher-Yates shuffle) into `perm`
    fn permutation(&mut self, perm: &mut [usize]) {
        for (i, v) in perm.iter_mut().enumerate() {
            *v = i;
        }
        for i in (1..perm.len()).rev() {
            let j = self.interval(i as u32) as usize;
            perm.swap(i, j);
        }
    }
}

fn binom(n: usize, k: usize) -> f64 {
    // scipy.special.binom for integer args (exact for the sizes used here)
    let k = k.min(n - k);
    let mut r = 1.0f64;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r.round()
}

/// Indices of features whose value in x differs from some background row (np.allclose, equal_nan).
fn varying(x: &[f64], bg: &[f64], n: usize, p: usize, groups: &Groups) -> Vec<usize> {
    let mut out = Vec::new();
    for g in 0..groups.len() {
        let mut differs = false;
        for &i in groups.members(g) {
        let a = x[i];
        if differs {
            break;
        }
        for j in 0..n {
            let b = bg[j * p + i];
            let close = if a.is_nan() || b.is_nan() {
                a.is_nan() && b.is_nan()
            } else if a.is_infinite() || b.is_infinite() {
                a == b
            } else {
                (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
            };
            if !close {
                differs = true;
                break;
            }
        }
        }
        if differs {
            out.push(g);
        }
    }
    out
}

/// Feature groups (DenseData.groups) as CSR; the default is one group per feature.
pub struct Groups {
    offs: Vec<usize>,
    idx: Vec<usize>,
}
impl Groups {
    fn singletons(p: usize) -> Groups {
        Groups { offs: (0..=p).collect(), idx: (0..p).collect() }
    }
    fn len(&self) -> usize {
        self.offs.len() - 1
    }
    fn members(&self, g: usize) -> &[usize] {
        &self.idx[self.offs[g]..self.offs[g + 1]]
    }
}

pub struct Plan {
    pub zero_tail: bool, // sampled-weight normalisation divided by an empty sum (numpy RuntimeWarning in shap)
    pub nsamples: usize,
    pub n_added: usize,
    pub masks: Vec<f64>, // nsamples x M
    pub kw: Vec<f64>,    // nsamples
}

/// Coalition plan for M varying features, exactly as KernelExplainer.explain builds it.
fn plan(m: usize, nsamples_req: Option<usize>, rng: &mut Mt) -> Plan {
    let mut nsamples = nsamples_req.unwrap_or(2 * m + 2048);
    if m <= 30 {
        let max_samples = (1usize << m) - 2;
        if nsamples > max_samples {
            nsamples = max_samples;
        }
    }
    let mut masks = vec![0.0f64; nsamples * m];
    let mut kw = vec![0.0f64; nsamples];
    let mut added = 0usize;
    let mut add = |mask: &[f64], w: f64, added: &mut usize, masks: &mut Vec<f64>, kw: &mut Vec<f64>| {
        masks[*added * m..(*added + 1) * m].copy_from_slice(mask);
        kw[*added] = w;
        *added += 1;
    };

    let num_subset_sizes = (m - 1).div_ceil(2);
    let num_paired = (m - 1) / 2;
    let mut weight_vector: Vec<f64> = (1..=num_subset_sizes).map(|i| (m as f64 - 1.0) / (i * (m - i)) as f64).collect();
    for w in weight_vector.iter_mut().take(num_paired) {
        *w *= 2.0;
    }
    let s: f64 = np_sum(&weight_vector);
    for w in weight_vector.iter_mut() {
        *w /= s;
    }

    let mut num_full_subsets = 0usize;
    let mut num_samples_left = nsamples as f64;
    let mut mask = vec![0.0f64; m];
    let mut rem = weight_vector.clone();
    for subset_size in 1..=num_subset_sizes {
        let mut nsubsets = binom(m, subset_size);
        if subset_size <= num_paired {
            nsubsets *= 2.0;
        }
        if num_samples_left * rem[subset_size - 1] / nsubsets >= 1.0 - 1e-8 {
            num_full_subsets += 1;
            num_samples_left -= nsubsets;
            if rem[subset_size - 1] < 1.0 {
                let d = 1.0 - rem[subset_size - 1];
                for r in rem.iter_mut() {
                    *r /= d;
                }
            }
            let mut w = weight_vector[subset_size - 1] / binom(m, subset_size);
            if subset_size <= num_paired {
                w /= 2.0;
            }
            // itertools.combinations(range(m), subset_size), lexicographic
            let mut idx: Vec<usize> = (0..subset_size).collect();
            loop {
                mask.fill(0.0);
                for &i in &idx {
                    mask[i] = 1.0;
                }
                add(&mask, w, &mut added, &mut masks, &mut kw);
                if subset_size <= num_paired {
                    for v in mask.iter_mut() {
                        *v = (*v - 1.0).abs();
                    }
                    add(&mask, w, &mut added, &mut masks, &mut kw);
                }
                // next combination
                let k = subset_size;
                let mut i = k;
                let mut found = false;
                while i > 0 {
                    i -= 1;
                    if idx[i] != i + m - k {
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
            }
        } else {
            break;
        }
    }

    let nfixed = added;
    let mut zero_tail = false;
    let mut samples_left = nsamples as i64 - added as i64;
    if num_full_subsets != num_subset_sizes {
        let mut rwv = weight_vector.clone();
        for r in rwv.iter_mut().take(num_paired) {
            *r /= 2.0;
        }
        let mut rwv: Vec<f64> = rwv[num_full_subsets..].to_vec();
        let s = np_sum(&rwv);
        for r in rwv.iter_mut() {
            *r /= s;
        }
        // np.random.choice(len(rwv), 4 * samples_left, p=rwv): cdf = cumsum(p) / cdf[-1], searchsorted right
        let mut cdf = Vec::with_capacity(rwv.len());
        let mut acc = 0.0;
        for r in &rwv {
            acc += r;
            cdf.push(acc);
        }
        let last = *cdf.last().unwrap();
        for c in cdf.iter_mut() {
            *c /= last;
        }
        let n_draws = 4 * samples_left.max(0) as usize;
        let ind_set: Vec<usize> = (0..n_draws)
            .map(|_| {
                let u = rng.next_double();
                cdf.iter().take_while(|&&c| c <= u).count()
            })
            .collect();
        let mut used: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut perm: Vec<usize> = (0..m).collect();
        let mut pos = 0usize;
        while samples_left > 0 && pos < n_draws {
            mask.fill(0.0);
            let ind = ind_set[pos];
            pos += 1;
            let subset_size = ind + num_full_subsets + 1;
            rng.permutation(&mut perm);
            for &pi in &perm[..subset_size] {
                mask[pi] = 1.0;
            }
            let key: Vec<u8> = mask.iter().map(|&v| v as u8).collect();
            let mut new_sample = false;
            let slot = match used.get(&key) {
                Some(&s) => {
                    kw[s] += 1.0;
                    s
                }
                None => {
                    new_sample = true;
                    used.insert(key, added);
                    samples_left -= 1;
                    let s = added;
                    add(&mask, 1.0, &mut added, &mut masks, &mut kw);
                    s
                }
            };
            if samples_left > 0 && subset_size <= num_paired {
                for v in mask.iter_mut() {
                    *v = (*v - 1.0).abs();
                }
                if new_sample {
                    samples_left -= 1;
                    add(&mask, 1.0, &mut added, &mut masks, &mut kw);
                } else {
                    kw[slot + 1] += 1.0;
                }
            }
        }
        let weight_left = np_sum(&weight_vector[num_full_subsets..]);
        let tail_sum = np_sum(&kw[nfixed..]);
        zero_tail = tail_sum == 0.0;
        for w in kw[nfixed..].iter_mut() {
            *w *= weight_left / tail_sum;
        }
    }
    Plan { zero_tail, nsamples, n_added: added, masks, kw }
}

/// numpy-style pairwise summation (matches np.sum for contiguous float64 to the last bit for n<=8,
/// and is close enough otherwise).
fn np_sum(a: &[f64]) -> f64 {
    fn pw(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut s = 0.0;
            for &v in a {
                s += v;
            }
            s
        } else if n <= 128 {
            let mut r = [0.0f64; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i + 8 <= n {
                for k in 0..8 {
                    r[k] += a[i + k];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pw(&a[..n2]) + pw(&a[n2..])
        }
    }
    pw(a)
}

// ------------------------------------------------------------------------------------------------
// sklearn.linear_model.lars_path(X, y, max_iter=r) with method="lar", Gram=None: only the active set
// (in order of entry) is needed by KernelExplainer.

fn min_pos(v: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::MAX;
    for x in v {
        if x > 0.0 && x < m {
            m = x;
        }
    }
    m
}

fn np_around15(x: f64) -> f64 {
    let f = 1e15f64;
    (x * f).round_ties_even() / f
}

/// X: n x p row-major. Returns the active set.
fn lars_active(x_in: &[f64], y: &[f64], n: usize, p: usize, max_iter: usize) -> Vec<usize> {
    // column-major copy (X.copy("F"))
    let mut cols: Vec<Vec<f64>> = (0..p).map(|j| (0..n).map(|i| x_in[i * p + j]).collect()).collect();
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    // Cov = X.T @ y: numpy's gemv accumulates with FMA in row order
    let mut cov_full: Vec<f64> = cols.iter().map(|c| c.iter().zip(y).fold(0.0, |acc, (a, b)| a.mul_add(*b, acc))).collect();
    let mut start = 0usize; // Cov = cov_full[start..]
    let max_features = max_iter.min(p);
    let mut l = vec![0.0f64; max_features.max(1) * max_features.max(1)];
    let ld = max_features.max(1);
    let mut indices: Vec<usize> = (0..p).collect();
    let mut active: Vec<usize> = Vec::new();
    let mut sign_active = vec![0i8; max_features.max(1)];
    let mut coef = vec![0.0f64; p];
    let (mut n_iter, mut n_active) = (0usize, 0usize);
    let tiny32 = f32::MIN_POSITIVE as f64;
    let equality_tolerance = f32::EPSILON as f64;
    let eps = f64::EPSILON;
    let mut drop = false;

    loop {
        let cov = &cov_full[start..];
        let (c_idx, c_, c) = if !cov.is_empty() {
            let mut bi = 0;
            let mut bv = cov[0].abs();
            for (i, v) in cov.iter().enumerate() {
                if v.abs() > bv {
                    bv = v.abs();
                    bi = i;
                }
            }
            (bi, cov[bi], cov[bi].abs())
        } else {
            (0, 0.0, 0.0)
        };
        let alpha = c / n as f64;
        if alpha <= equality_tolerance {
            break;
        }
        if n_iter >= max_iter || n_active >= p {
            break;
        }
        if !drop {
            sign_active[n_active] = if c_ > 0.0 { 1 } else if c_ < 0.0 { -1 } else { 0 };
            let (mm, nn) = (n_active, c_idx + n_active);
            cov_full.swap(start + c_idx, start);
            indices.swap(nn, mm);
            start += 1;
            cols.swap(nn, mm);
            let cn = &cols[n_active];
            let c2 = {
                let nrm = cn.iter().map(|v| v * v).sum::<f64>().sqrt();
                nrm * nrm
            };
            for k in 0..n_active {
                l[n_active * ld + k] = dot(cn, &cols[k]);
            }
            if n_active > 0 {
                // forward substitution: L[:na,:na] z = L[na,:na]
                for i in 0..n_active {
                    let mut s = l[n_active * ld + i];
                    for k in 0..i {
                        s -= l[i * ld + k] * l[n_active * ld + k];
                    }
                    l[n_active * ld + i] = s / l[i * ld + i];
                }
            }
            let v: f64 = (0..n_active).map(|k| l[n_active * ld + k] * l[n_active * ld + k]).sum();
            let diag = (c2 - v).abs().sqrt().max(eps);
            l[n_active * ld + n_active] = diag;
            if diag < 1e-7 {
                lasso::sk_event(); // sklearn: ConvergenceWarning "Regressors in active set degenerate"
                // degenerate regressor: shortened Cov is undone, Cov[0] zeroed and swapped back
                start -= 1;
                cov_full[start] = 0.0;
                cov_full.swap(start + c_idx, start);
                continue;
            }
            active.push(indices[n_active]);
            n_active += 1;
        }

        // least_squares = (L L^T)^-1 sign_active
        let na = n_active;
        let mut ls: Vec<f64> = (0..na).map(|i| sign_active[i] as f64).collect();
        chol_solve(&l, ld, na, &mut ls);
        let aa;
        if na == 1 && ls[0] == 0.0 {
            ls[0] = 1.0;
            aa = 1.0;
        } else {
            let mut a = 1.0 / ls.iter().zip(&sign_active[..na]).map(|(v, s)| v * *s as f64).sum::<f64>().sqrt();
            if !a.is_finite() {
                lasso::sk_event();
                let mut i = 0;
                let mut l2 = vec![0.0f64; na * na];
                for r in 0..na {
                    for c in 0..na {
                        l2[r * na + c] = l[r * ld + c];
                    }
                }
                while !a.is_finite() {
                    for d in 0..na {
                        l2[d * na + d] += 2f64.powi(i) * eps;
                    }
                    ls = (0..na).map(|k| sign_active[k] as f64).collect();
                    chol_solve(&l2, na, na, &mut ls);
                    let tmp = ls.iter().zip(&sign_active[..na]).map(|(v, s)| v * *s as f64).sum::<f64>().max(eps);
                    a = 1.0 / tmp.sqrt();
                    i += 1;
                }
            }
            aa = a;
            for v in ls.iter_mut() {
                *v *= aa;
            }
        }
        // eq_dir = X[:, :na] @ ls ; corr_eq_dir = X[:, na:]^T eq_dir
        let mut eq_dir = vec![0.0f64; n];
        for k in 0..na {
            let ck = &cols[k];
            let lk = ls[k];
            for i in 0..n {
                eq_dir[i] += ck[i] * lk;
            }
        }
        let corr: Vec<f64> = (na..p).map(|j| np_around15(dot(&cols[j], &eq_dir))).collect();
        let cov = &cov_full[start..];
        if corr.iter().any(|cd| aa - cd + tiny32 == 0.0 || aa + cd + tiny32 == 0.0) || ls.iter().any(|l_| l_ + tiny32 == 0.0) {
            lasso::sk_event(); // numpy divide-by-zero RuntimeWarning in sklearn
        }
        let g1 = min_pos(cov.iter().zip(&corr).map(|(cv, cd)| (c - cv) / (aa - cd + tiny32)));
        let g2 = min_pos(cov.iter().zip(&corr).map(|(cv, cd)| (c + cv) / (aa + cd + tiny32)));
        let gamma = g1.min(g2).min(c / aa);
        drop = false;
        let z: Vec<f64> = active.iter().zip(&ls).map(|(&a, l_)| -coef[a] / (l_ + tiny32)).collect();
        let z_pos = min_pos(z.iter().copied());
        if z_pos < gamma {
            for (i, zv) in z.iter().enumerate().rev() {
                if *zv == z_pos {
                    sign_active[i] = -sign_active[i];
                }
            }
            drop = true;
        }
        n_iter += 1;
        let prev = coef.clone();
        coef.iter_mut().for_each(|v| *v = 0.0);
        for (k, &a) in active.iter().enumerate() {
            coef[a] = prev[a] + gamma * ls[k];
        }
        for (cv, cd) in cov_full[start..].iter_mut().zip(&corr) {
            *cv -= gamma * cd;
        }
    }
    active
}

/// Solve (L L^T) x = b in place, L lower-triangular with leading dimension ld.
fn chol_solve(l: &[f64], ld: usize, n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * ld + k] * b[k];
        }
        b[i] = s / l[i * ld + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * ld + i] * b[k];
        }
        b[i] = s / l[i * ld + i];
    }
}

/// Solve A x = b (A: n x n row-major) by Gaussian elimination with partial pivoting (LAPACK gesv).
fn gesv(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col] == 0.0 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    Some(b)
}

#[derive(Clone, Copy)]
enum Link {
    Identity,
    Logit,
}
impl Link {
    fn f(self, x: f64) -> f64 {
        match self {
            Link::Identity => x,
            Link::Logit => {
                // shap.utils._legacy.LogitLink.f: np.clip(x, 1e-15, 1 - 1e-15) first
                let eps = 1e-15;
                let xc = if x.is_nan() { x } else { x.max(eps).min(1.0 - eps) };
                (xc / (1.0 - xc)).ln()
            }
        }
    }
}

/// mask_aug (2ns x m, row-major) and eyAdj_aug of KernelExplainer.solve's feature selection.
fn l1_aug(ey_adj: &[f64], total: f64, masks: &[f64], kw: &[f64], ns: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let s: Vec<f64> = (0..ns).map(|i| masks[i * m..(i + 1) * m].iter().sum()).collect();
    let rows = 2 * ns;
    let mut xa = vec![0.0f64; rows * m];
    let mut ya = vec![0.0f64; rows];
    for i in 0..ns {
        let w1 = (kw[i] * (m as f64 - s[i])).sqrt();
        let w2 = (kw[i] * s[i]).sqrt();
        ya[i] = ey_adj[i] * w1;
        ya[ns + i] = (ey_adj[i] - total) * w2;
        for j in 0..m {
            let mv = masks[i * m + j];
            xa[i * m + j] = w1 * mv;
            xa[(ns + i) * m + j] = w2 * (mv - 1.0);
        }
    }
    (xa, ya)
}

/// The Rust ports of shap's l1 feature selection (lars_path / LassoLarsIC pipeline / Lasso). Returns the
/// selected columns or the ValueError sklearn raises, plus whether sklearn would have warned or raised
/// somewhere along the way (lasso::sk_event), in which case exact side effects need sklearn itself.
fn l1_select(xa: &[f64], ya: &[f64], rows: usize, m: usize, l1: L1) -> (Result<Vec<usize>, String>, bool) {
    lasso::take_sk_event();
    let r = match l1 {
        L1::NumFeatures(r) => Ok(lars_active(xa, ya, rows, m, r)),
        L1::Aic | L1::Bic => lasso::lasso_lars_ic(xa, ya, rows, m, matches!(l1, L1::Bic)).map_err(|_| {
            "You are using LassoLarsIC in the case where the number of samples is smaller than \
             the number of features. In this setting, getting a good estimate for the variance of the noise \
             is not possible. Provide an estimate of the noise variance in the constructor."
                .to_string()
        }),
        L1::Alpha(a) => lasso::lasso_cd(xa, ya, rows, m, a).ok_or_else(|| {
            "Coordinate descent iterations resulted in non-finite parameter values. The input \
             data may contain large values and need to be preprocessed."
                .to_string()
        }),
        _ => Ok((0..m).collect()),
    };
    let ev = lasso::take_sk_event() || r.is_err();
    (r, ev)
}

/// The constrained weighted least squares of KernelExplainer.solve on the selected columns `nonzero`
/// (non-empty). Returns phi over the M varying features and whether np.linalg.solve found the normal
/// equations singular (shap then warns and uses lstsq).
#[allow(clippy::too_many_arguments)]
fn wls(ey_adj: &[f64], total: f64, masks: &[f64], kw: &[f64], ns: usize, m: usize, nonzero: &[usize]) -> (Vec<f64>, bool) {
    let mut phi = vec![0.0f64; m];
    let last = *nonzero.last().unwrap();
    let rest = &nonzero[..nonzero.len() - 1];
    let k = rest.len();
    // y = eyAdj - mask[:, last] * total ; X = mask[:, rest] - mask[:, last]
    let mut xtwx = vec![0.0f64; k * k];
    let mut xtwy = vec![0.0f64; k];
    let mut xr = vec![0.0f64; k];
    for i in 0..ns {
        let ml = masks[i * m + last];
        let yi = ey_adj[i] - ml * total;
        for (a, &j) in rest.iter().enumerate() {
            xr[a] = masks[i * m + j] - ml;
        }
        let w = kw[i];
        for a in 0..k {
            let wxa = w * xr[a];
            if wxa != 0.0 {
                xtwy[a] += wxa * yi;
                for b in 0..k {
                    xtwx[b * k + a] += xr[b] * wxa;
                }
            }
        }
    }
    let mut singular = false;
    let wsol = match gesv(xtwx, xtwy, k) {
        Some(v) => v,
        None => {
            // singular normal equations: shap falls back to np.linalg.lstsq(sqrt(W) X, sqrt(W) y)
            singular = true;
            let mut a = vec![0.0f64; ns * k];
            let mut b = vec![0.0f64; ns];
            for i in 0..ns {
                let sw = kw[i].sqrt();
                let ml = masks[i * m + last];
                b[i] = sw * (ey_adj[i] - ml * total);
                for (c, &j) in rest.iter().enumerate() {
                    a[i * k + c] = sw * (masks[i * m + j] - ml);
                }
            }
            lasso::lstsq(&a, &b, ns, k)
        }
    };
    let mut sumw = 0.0;
    for (a, &j) in rest.iter().enumerate() {
        phi[j] = wsol[a];
        sumw += wsol[a];
    }
    phi[last] = total - sumw;
    for v in phi.iter_mut() {
        if v.abs() < 1e-10 {
            *v = 0.0;
        }
    }
    (phi, singular)
}

/// KernelExplainer.solve for one output dimension (batched standalone API). Returns phi over the M varying
/// features.
#[allow(clippy::too_many_arguments)]
fn solve(
    ey: &[f64],
    masks: &[f64],
    kw: &[f64],
    ns: usize,
    m: usize,
    fx: f64,
    fnull: f64,
    link: Link,
    l1: L1,
    selector: Option<&Py<PyAny>>,
) -> Result<Vec<f64>, PyErr> {
    let lf_null = link.f(fnull);
    let ey_adj: Vec<f64> = ey.iter().map(|&v| link.f(v) - lf_null).collect();
    let total = link.f(fx) - lf_null;
    let mut nonzero: Vec<usize> = (0..m).collect();
    // shap: `l1_reg == "auto"` uses "aic" when fraction_evaluated < 0.2, else no selection
    let max_samples = if m <= 30 { ((1u64 << m) - 2) as f64 } else { (1u64 << 30) as f64 };
    let l1 = match l1 {
        L1::Auto if (ns as f64) / max_samples < 0.2 => L1::Aic,
        L1::Auto => L1::Off,
        o => o,
    };
    if !matches!(l1, L1::Off) {
        let rows = 2 * ns;
        let (xa, ya) = l1_aug(&ey_adj, total, masks, kw, ns, m);
        nonzero = if let Some(sel) = selector {
            // Off arm64 the LARS/Lasso selection is delegated to sklearn itself (what shap calls), on the same
            // bit-identical mask_aug/eyAdj_aug: exact LARS ties depend on the BLAS summation order of the platform.
            let (code, param) = l1.code();
            Python::attach(|py| -> PyResult<Vec<usize>> {
                let xn = PyArray1::from_vec(py, xa).reshape([rows, m])?;
                let yn = PyArray1::from_vec(py, ya);
                sel.bind(py).call1((xn, yn, code, param))?.extract::<Vec<usize>>()
            })?
        } else {
            l1_select(&xa, &ya, rows, m, l1).0.map_err(pyo3::exceptions::PyValueError::new_err)?
        };
    }
    if nonzero.is_empty() {
        return Ok(vec![0.0f64; m]);
    }
    Ok(wls(&ey_adj, total, masks, kw, ns, m, &nonzero).0)
}

/// l1_reg modes of KernelExplainer.solve
#[derive(Clone, Copy)]
pub enum L1 {
    Off,
    NumFeatures(usize),
    Aic,
    Bic,
    Alpha(f64),
    Auto,
}
impl L1 {
    fn code(self) -> (u32, f64) {
        match self {
            L1::NumFeatures(r) => (1, r as f64),
            L1::Aic => (2, 0.0),
            L1::Bic => (3, 0.0),
            L1::Alpha(a) => (4, a),
            L1::Auto => (5, 0.0),
            L1::Off => (0, 0.0),
        }
    }
    fn from_code(code: u32, param: f64) -> L1 {
        match code {
            1 => L1::NumFeatures(param as usize),
            2 => L1::Aic,
            3 => L1::Bic,
            4 => L1::Alpha(param),
            5 => L1::Auto,
            _ => L1::Off,
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Python bindings

// ------------------------------------------------------------------------------------------------
// Batched API: plans for many rows built in parallel, synthetic data written into one buffer, and
// the solves run in parallel. Python only calls the user's model in between.

struct RowPlan {
    var: Vec<usize>,
    plan: Plan,
    start: usize, // first synthetic row in the batch buffer
}

#[pyclass]
pub struct KernelBatch {
    rows: Vec<RowPlan>,
    total_rows: usize,
}

#[pymethods]
impl KernelBatch {
    #[getter]
    fn total_rows(&self) -> usize {
        self.total_rows
    }
    /// synthetic rows per explained row (n_added * N), in order
    #[getter]
    fn row_sizes(&self) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::with_capacity(self.rows.len());
        for (i, r) in self.rows.iter().enumerate() {
            let end = self.rows.get(i + 1).map(|n| n.start).unwrap_or(self.total_rows);
            v.push(end - r.start);
        }
        v
    }
}

/// Coalition plans for every row, in row order (consumes the RNG exactly like shap's per-row explain).
fn make_plans(vars: Vec<Vec<usize>>, nsamples: Option<usize>, rng: &mut Mt, extra_draws: usize, n: usize) -> (Vec<RowPlan>, usize) {
    let mut rows: Vec<RowPlan> = vars
        .into_iter()
        .map(|var| {
            let m = var.len();
            let plan = if m >= 2 {
                let pl = plan(m, nsamples, rng);
                for _ in 0..extra_draws {
                    rng.interval(2147483646);
                }
                pl
            } else {
                Plan { zero_tail: false, nsamples: 0, n_added: 0, masks: vec![], kw: vec![] }
            };
            RowPlan { var, plan, start: 0 }
        })
        .collect();
    let mut total = 0;
    for r in rows.iter_mut() {
        r.start = total;
        total += r.plan.n_added * n;
    }
    (rows, total)
}

/// CSR matrix view (sorted column indices per row).
struct Csr<'a> {
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f64],
}
impl Csr<'_> {
    fn row(&self, i: usize) -> (&[i64], &[f64]) {
        let (a, b) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
        (&self.indices[a..b], &self.data[a..b])
    }
}

/// shap's KernelExplainer.varying_groups for sparse x and background (one group per feature).
fn varying_sparse(xi: &[i64], xv: &[f64], bg_cols: &[Vec<f64>], bg_nz_cols: &[usize], n: usize) -> Vec<usize> {
    // candidates: union of nonzero columns of the background and of x (sorted, unique)
    let mut cand: Vec<usize> = bg_nz_cols.to_vec();
    cand.extend(xi.iter().zip(xv).filter(|(_, &v)| v != 0.0).map(|(&c, _)| c as usize));
    cand.sort_unstable();
    cand.dedup();
    let xval = |c: usize| -> f64 {
        match xi.binary_search(&(c as i64)) {
            Ok(k) => xv[k],
            Err(_) => 0.0,
        }
    };
    cand.into_iter()
        .filter(|&c| {
            let nz = &bg_cols[c];
            if nz.is_empty() {
                return true;
            }
            let x = xval(c);
            let mismatches = nz.iter().filter(|&&v| (v - x).abs() > 1e-7).count();
            !(mismatches == 0 && !(x.abs() > 1e-7 && nz.len() < n))
        })
        .collect()
}

/// kernel_prepare_batch_sparse(X CSR (k rows), bg CSR (N rows), P, nsamples, mt_key, mt_pos, extra_draws)
///   -> (KernelBatch, synth_indptr, synth_indices, synth_data, mt_key, mt_pos). Same as kernel_prepare_batch
/// for scipy.sparse inputs; the synthetic rows are returned as CSR (background row with the coalition's
/// columns set to x's values, zeros dropped).
#[pyfunction]
#[pyo3(signature = (x_indptr, x_indices, x_data, bg_indptr, bg_indices, bg_data, p, nsamples, mt_key, mt_pos, extra_draws))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn kernel_prepare_batch_sparse<'py>(
    py: Python<'py>,
    x_indptr: PyReadonlyArray1<i64>,
    x_indices: PyReadonlyArray1<i64>,
    x_data: PyReadonlyArray1<f64>,
    bg_indptr: PyReadonlyArray1<i64>,
    bg_indices: PyReadonlyArray1<i64>,
    bg_data: PyReadonlyArray1<f64>,
    p: usize,
    nsamples: Option<usize>,
    mt_key: PyReadonlyArray1<u32>,
    mt_pos: usize,
    extra_draws: usize,
) -> PyResult<(
    KernelBatch,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u32>>,
    usize,
)> {
    use rayon::prelude::*;
    let x = Csr { indptr: x_indptr.as_slice()?, indices: x_indices.as_slice()?, data: x_data.as_slice()? };
    let bg = Csr { indptr: bg_indptr.as_slice()?, indices: bg_indices.as_slice()?, data: bg_data.as_slice()? };
    let k = x.indptr.len() - 1;
    let n = bg.indptr.len() - 1;
    let mut rng = Mt { key: mt_key.as_slice()?.to_vec(), pos: mt_pos };
    // background columns: nonzero values per column
    let mut bg_cols: Vec<Vec<f64>> = vec![Vec::new(); p];
    for r in 0..n {
        let (ci, cv) = bg.row(r);
        for (&c, &v) in ci.iter().zip(cv) {
            if v != 0.0 {
                bg_cols[c as usize].push(v);
            }
        }
    }
    let bg_nz_cols: Vec<usize> = (0..p).filter(|&c| !bg_cols[c].is_empty()).collect();
    let vars: Vec<Vec<usize>> = (0..k)
        .into_par_iter()
        .map(|i| {
            let (xi, xv) = x.row(i);
            varying_sparse(xi, xv, &bg_cols, &bg_nz_cols, n)
        })
        .collect();
    let (rows, total) = make_plans(vars, nsamples, &mut rng, extra_draws, n);
    // per explained row: CSR pieces (row lengths, indices, data) built in parallel
    let pieces: Vec<(Vec<i64>, Vec<i64>, Vec<f64>)> = rows
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            let (xi, xv) = x.row(i);
            let xval = |c: usize| -> f64 {
                match xi.binary_search(&(c as i64)) {
                    Ok(q) => xv[q],
                    Err(_) => 0.0,
                }
            };
            let m = r.var.len();
            let (mut lens, mut ind, mut dat) = (Vec::new(), Vec::new(), Vec::new());
            let mut patch: Vec<(i64, f64)> = Vec::with_capacity(m);
            for s in 0..r.plan.n_added {
                let mk = &r.plan.masks[s * m..(s + 1) * m];
                patch.clear();
                for (jj, &c) in r.var.iter().enumerate() {
                    if mk[jj] == 1.0 {
                        patch.push((c as i64, xval(c)));
                    }
                }
                for row in 0..n {
                    let (ci, cv) = bg.row(row);
                    let before = ind.len();
                    let (mut a, mut b) = (0usize, 0usize);
                    while a < ci.len() || b < patch.len() {
                        if b == patch.len() || (a < ci.len() && ci[a] < patch[b].0) {
                            ind.push(ci[a]);
                            dat.push(cv[a]);
                            a += 1;
                        } else {
                            if a < ci.len() && ci[a] == patch[b].0 {
                                a += 1;
                            }
                            if patch[b].1 != 0.0 {
                                ind.push(patch[b].0);
                                dat.push(patch[b].1);
                            }
                            b += 1;
                        }
                    }
                    lens.push((ind.len() - before) as i64);
                }
            }
            (lens, ind, dat)
        })
        .collect();
    let nnz: usize = pieces.iter().map(|p| p.1.len()).sum();
    let mut indptr = Vec::with_capacity(total + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0i64);
    let mut acc = 0i64;
    for (lens, ind, dat) in pieces {
        for l in lens {
            acc += l;
            indptr.push(acc);
        }
        indices.extend(ind);
        data.extend(dat);
    }
    Ok((
        KernelBatch { rows, total_rows: total },
        Array1::from(indptr).into_pyarray(py),
        Array1::from(indices).into_pyarray(py),
        Array1::from(data).into_pyarray(py),
        Array1::from(rng.key).into_pyarray(py),
        rng.pos,
    ))
}

/// kernel_prepare_batch(X (k x P), bg (N x P), nsamples, mt_key, mt_pos, extra_draws)
///   -> (KernelBatch, synth (total x P), mt_key, mt_pos)
/// The coalition plans consume numpy's global MT19937 stream (state passed in and returned) in the same
/// order as shap: row by row, sampling draws, then `extra_draws` randint(0, 2**31-1) calls per row that
/// reaches the solve (sklearn's Lasso seeds its coordinate descent from the global RNG once per output).
#[pyfunction]
#[pyo3(signature = (x, bg, nsamples, mt_key, mt_pos, extra_draws, group_offs=None, group_idx=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn kernel_prepare_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    bg: PyReadonlyArray2<f64>,
    nsamples: Option<usize>,
    mt_key: PyReadonlyArray1<u32>,
    mt_pos: usize,
    extra_draws: usize,
    group_offs: Option<PyReadonlyArray1<i64>>,
    group_idx: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(KernelBatch, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<u32>>, usize)> {
    use rayon::prelude::*;
    let (k, p) = (x.shape()[0], x.shape()[1]);
    let n = bg.shape()[0];
    let xs = x.as_slice()?.to_vec();
    let bgs = bg.as_slice()?.to_vec();
    let mut rng = Mt { key: mt_key.as_slice()?.to_vec(), pos: mt_pos };
    let groups = match (group_offs, group_idx) {
        (Some(o), Some(ix)) => Groups {
            offs: o.as_slice()?.iter().map(|&v| v as usize).collect(),
            idx: ix.as_slice()?.iter().map(|&v| v as usize).collect(),
        },
        _ => Groups::singletons(p),
    };
    let (rows, synth, total, rng) = py.detach(|| {
        let vars: Vec<Vec<usize>> = (0..k).into_par_iter().map(|i| varying(&xs[i * p..(i + 1) * p], &bgs, n, p, &groups)).collect();
        let (rows, total) = make_plans(vars, nsamples, &mut rng, extra_draws, n);
        let mut synth = vec![0.0f64; total * p];
        // split the buffer per row and fill in parallel
        let mut chunks: Vec<&mut [f64]> = Vec::with_capacity(k);
        let mut rest: &mut [f64] = &mut synth;
        for r in rows.iter() {
            let (a, b) = rest.split_at_mut(r.plan.n_added * n * p);
            chunks.push(a);
            rest = b;
        }
        chunks.into_par_iter().zip(rows.par_iter()).enumerate().for_each(|(i, (buf, r))| {
            let xr = &xs[i * p..(i + 1) * p];
            let m = r.var.len();
            let mut patch: Vec<(usize, f64)> = Vec::with_capacity(p);
            for s in 0..r.plan.n_added {
                let block = &mut buf[s * n * p..(s + 1) * n * p];
                let mk = &r.plan.masks[s * m..(s + 1) * m];
                patch.clear();
                for (jj, &g) in r.var.iter().enumerate() {
                    if mk[jj] == 1.0 {
                        for &f in groups.members(g) {
                            patch.push((f, xr[f]));
                        }
                    }
                }
                // background row with the coalition's features taken from x (row by row, stays in cache)
                for (dst, src) in block.chunks_exact_mut(p).zip(bgs.chunks_exact(p)) {
                    dst.copy_from_slice(src);
                    for &(f, xv) in &patch {
                        dst[f] = xv;
                    }
                }
            }
        });
        (rows, synth, total, rng)
    });
    Ok((
        KernelBatch { rows, total_rows: total },
        Array2::from_shape_vec((total, p), synth).unwrap().into_pyarray(py),
        Array1::from(rng.key).into_pyarray(py),
        rng.pos,
    ))
}

/// kernel_solve_batch(batch, y (total x D), bg_weights, fx (k x D), fnull (D), P, link, l1_code, l1_param)
///   -> phi (k x P x D). l1_code: 0 off, 1 num_features(param), 2 aic, 3 bic, 4 Lasso(alpha=param), 5 auto.
#[pyfunction]
#[pyo3(signature = (batch, y, bg_weights, fx, fnull, p, link, l1_code, l1_param, selector=None))]
#[allow(clippy::too_many_arguments)]
fn kernel_solve_batch<'py>(
    py: Python<'py>,
    batch: &KernelBatch,
    y: PyReadonlyArray2<f64>,
    bg_weights: PyReadonlyArray1<f64>,
    fx: PyReadonlyArray2<f64>,
    fnull: PyReadonlyArray1<f64>,
    p: usize,
    link: u32,
    l1_code: u32,
    l1_param: f64,
    selector: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let l1 = L1::from_code(l1_code, l1_param);
    use rayon::prelude::*;
    let link = if link == 1 { Link::Logit } else { Link::Identity };
    let d = y.shape()[1];
    let ys = y.as_slice()?;
    let w = bg_weights.as_slice()?;
    let n = w.len();
    let fxs = fx.as_slice()?;
    let fns = fnull.as_slice()?;
    let k = batch.rows.len();
    let mut phi = vec![0.0f64; k * p * d];
    let err: std::sync::Mutex<Option<PyErr>> = std::sync::Mutex::new(None);
    let selector = selector.as_ref();
    py.detach(|| {
        phi.par_chunks_mut(p * d).zip(batch.rows.par_iter()).enumerate().for_each(|(i, (ph, r))| {
            let m = r.var.len();
            let fxr = &fxs[i * d..(i + 1) * d];
            if m == 1 {
                for kk in 0..d {
                    ph[r.var[0] * d + kk] = link.f(fxr[kk]) - link.f(fns[kk]);
                }
            } else if m >= 2 {
                let ns = r.plan.nsamples;
                let mut ey = vec![0.0f64; ns * d];
                for s in 0..r.plan.n_added {
                    for j in 0..n {
                        let yrow = (r.start + s * n + j) * d;
                        for kk in 0..d {
                            ey[s * d + kk] = madd(ys[yrow + kk], w[j], ey[s * d + kk]);
                        }
                    }
                }
                for kk in 0..d {
                    let eyk: Vec<f64> = (0..ns).map(|s| ey[s * d + kk]).collect();
                    match solve(&eyk, &r.plan.masks, &r.plan.kw, ns, m, fxr[kk], fns[kk], link, l1, selector) {
                        Ok(v) => {
                            for (jj, &f) in r.var.iter().enumerate() {
                                ph[f * d + kk] = v[jj];
                            }
                        }
                        Err(e) => {
                            err.lock().unwrap().get_or_insert(e);
                        }
                    }
                }
            }
        });
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(numpy::ndarray::Array3::from_shape_vec((k, p, d), phi).unwrap().into_pyarray(py))
}

// ------------------------------------------------------------------------------------------------
// Per-row API used by the shap patch: shap's own `shap_values` loops over rows and calls `explain`, which
// runs these steps in shap's order, with the user's model (and anything it does with np.random) called
// from Python in between, exactly where shap calls it.

fn groups_from(offs: Option<PyReadonlyArray1<i64>>, idx: Option<PyReadonlyArray1<i64>>, p: usize) -> PyResult<Groups> {
    Ok(match (offs, idx) {
        (Some(o), Some(ix)) => Groups {
            offs: o.as_slice()?.iter().map(|&v| v as usize).collect(),
            idx: ix.as_slice()?.iter().map(|&v| v as usize).collect(),
        },
        _ => Groups::singletons(p),
    })
}

/// KernelExplainer.varying_groups for a dense numeric row x (P,) and dense background (N x P).
#[pyfunction]
#[pyo3(signature = (x, bg, group_offs=None, group_idx=None))]
fn kernel_row_varying(x: PyReadonlyArray1<f64>, bg: PyReadonlyArray2<f64>, group_offs: Option<PyReadonlyArray1<i64>>,
                      group_idx: Option<PyReadonlyArray1<i64>>) -> PyResult<Vec<usize>> {
    let (n, p) = (bg.shape()[0], bg.shape()[1]);
    let groups = groups_from(group_offs, group_idx, p)?;
    Ok(varying(x.as_slice()?, bg.as_slice()?, n, p, &groups))
}

/// KernelExplainer.varying_groups for a sparse row (sorted CSR column indices / values) and CSR background.
#[pyfunction]
fn kernel_row_varying_sparse(x_indices: PyReadonlyArray1<i64>, x_data: PyReadonlyArray1<f64>, bg_indptr: PyReadonlyArray1<i64>,
                             bg_indices: PyReadonlyArray1<i64>, bg_data: PyReadonlyArray1<f64>, p: usize) -> PyResult<Vec<usize>> {
    let bg = Csr { indptr: bg_indptr.as_slice()?, indices: bg_indices.as_slice()?, data: bg_data.as_slice()? };
    let n = bg.indptr.len() - 1;
    let mut bg_cols: Vec<Vec<f64>> = vec![Vec::new(); p];
    for r in 0..n {
        let (ci, cv) = bg.row(r);
        for (&c, &v) in ci.iter().zip(cv) {
            if v != 0.0 {
                bg_cols[c as usize].push(v);
            }
        }
    }
    let bg_nz_cols: Vec<usize> = (0..p).filter(|&c| !bg_cols[c].is_empty()).collect();
    Ok(varying_sparse(x_indices.as_slice()?, x_data.as_slice()?, &bg_cols, &bg_nz_cols, n))
}

/// Coalitions for M >= 2 varying groups and the (already clamped) nsamples, drawing from numpy's global
/// MT19937 state (mt_key, mt_pos) exactly like explain's sampling loop.
///   -> (maskMatrix (nsamples x M), kernelWeights, nsamplesAdded, mt_key, mt_pos, zero_tail)
#[pyfunction]
#[allow(clippy::type_complexity)]
fn kernel_row_plan<'py>(py: Python<'py>, m: usize, nsamples: usize, mt_key: PyReadonlyArray1<u32>, mt_pos: usize)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, usize, Bound<'py, PyArray1<u32>>, usize, bool)> {
    let mut rng = Mt { key: mt_key.as_slice()?.to_vec(), pos: mt_pos };
    let pl = py.detach(|| plan(m, Some(nsamples), &mut rng));
    Ok((
        Array2::from_shape_vec((pl.nsamples, m), pl.masks).unwrap().into_pyarray(py),
        Array1::from(pl.kw).into_pyarray(py),
        pl.n_added,
        Array1::from(rng.key).into_pyarray(py),
        rng.pos,
        pl.zero_tail,
    ))
}

fn coalition_patch(mk: &[f64], vg: &Groups, xr: &[f64], out: &mut Vec<(usize, f64)>) {
    out.clear();
    for (jj, &mv) in mk.iter().enumerate() {
        if mv == 1.0 {
            for &f in vg.members(jj) {
                out.push((f, xr[f]));
            }
        }
    }
    out.sort_by_key(|e| e.0);
    out.dedup_by_key(|e| e.0);
}

/// shap's synth_data after allocate + addsample: nsamples tiles of the background (N x P), tile s with the
/// columns of the varying groups selected by mask s (vg = members of each varying group) set to x.
/// Tiles past n_added stay background, as in shap.
#[pyfunction]
fn kernel_row_synth<'py>(py: Python<'py>, x: PyReadonlyArray1<f64>, bg: PyReadonlyArray2<f64>, masks: PyReadonlyArray2<f64>,
                         n_added: usize, vg_offs: PyReadonlyArray1<i64>, vg_idx: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use rayon::prelude::*;
    let (n, p) = (bg.shape()[0], bg.shape()[1]);
    let (ns, m) = (masks.shape()[0], masks.shape()[1]);
    let vg = groups_from(Some(vg_offs), Some(vg_idx), p)?;
    let (xr, bgs, mk) = (x.as_slice()?, bg.as_slice()?, masks.as_slice()?);
    let mut synth = vec![0.0f64; ns * n * p];
    py.detach(|| {
        synth.par_chunks_mut((n * p).max(1)).enumerate().for_each_init(Vec::new, |patch, (s, block)| {
            if s < n_added {
                coalition_patch(&mk[s * m..(s + 1) * m], &vg, xr, patch);
            } else {
                patch.clear();
            }
            for (dst, src) in block.chunks_exact_mut(p.max(1)).zip(bgs.chunks_exact(p.max(1))) {
                dst.copy_from_slice(src);
                for &(f, xv) in patch.iter() {
                    dst[f] = xv;
                }
            }
        });
    });
    Ok(Array2::from_shape_vec((ns * n, p), synth).unwrap().into_pyarray(py))
}

/// Same for a CSR background (sorted, duplicate-free): returns the CSR parts of the nsamples * N synthetic
/// rows, matching the lil_matrix shap builds (background entries kept as stored, coalition columns set to
/// x's value; a zero assignment removes the entry like lil's __setitem__).
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn kernel_row_synth_sparse<'py>(py: Python<'py>, x: PyReadonlyArray1<f64>, bg_indptr: PyReadonlyArray1<i64>, bg_indices: PyReadonlyArray1<i64>,
                                bg_data: PyReadonlyArray1<f64>, masks: PyReadonlyArray2<f64>, n_added: usize,
                                vg_offs: PyReadonlyArray1<i64>, vg_idx: PyReadonlyArray1<i64>)
    -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let bg = Csr { indptr: bg_indptr.as_slice()?, indices: bg_indices.as_slice()?, data: bg_data.as_slice()? };
    let n = bg.indptr.len() - 1;
    let xr = x.as_slice()?;
    let p = xr.len();
    let (ns, m) = (masks.shape()[0], masks.shape()[1]);
    let vg = groups_from(Some(vg_offs), Some(vg_idx), p)?;
    let mk = masks.as_slice()?;
    let mut indptr: Vec<i64> = Vec::with_capacity(ns * n + 1);
    let mut ind: Vec<i64> = Vec::new();
    let mut dat: Vec<f64> = Vec::new();
    indptr.push(0);
    let mut patch: Vec<(usize, f64)> = Vec::new();
    for s in 0..ns {
        if s < n_added {
            coalition_patch(&mk[s * m..(s + 1) * m], &vg, xr, &mut patch);
        } else {
            patch.clear();
        }
        for row in 0..n {
            let (ci, cv) = bg.row(row);
            let (mut a, mut b) = (0usize, 0usize);
            while a < ci.len() || b < patch.len() {
                if b == patch.len() || (a < ci.len() && (ci[a] as usize) < patch[b].0) {
                    ind.push(ci[a]);
                    dat.push(cv[a]);
                    a += 1;
                } else {
                    if a < ci.len() && ci[a] as usize == patch[b].0 {
                        a += 1;
                    }
                    if patch[b].1 != 0.0 {
                        ind.push(patch[b].0 as i64);
                        dat.push(patch[b].1);
                    }
                    b += 1;
                }
            }
            indptr.push(ind.len() as i64);
        }
    }
    Ok((Array1::from(indptr).into_pyarray(py), Array1::from(ind).into_pyarray(py), Array1::from(dat).into_pyarray(py)))
}

/// `acc + a * b` the way shap's C++ (`eyVal[k] += y(i * N + j, k) * weights(j)`) is compiled: C/C++ compilers
/// contract it to a fused multiply-add on aarch64 (FMA is baseline there), not on baseline x86-64.
#[inline]
fn madd(a: f64, b: f64, acc: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, acc)
    } else {
        acc + a * b
    }
}

/// shap's compute_exp_val: ey[i] = sum_j y[i*N + j] * w[j] for the n_added evaluated tiles (rest zero).
#[pyfunction]
fn kernel_row_ey<'py>(py: Python<'py>, y: PyReadonlyArray2<f64>, w: PyReadonlyArray1<f64>, nsamples: usize, n_added: usize)
    -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d = y.shape()[1];
    let (ys, ws) = (y.as_slice()?, w.as_slice()?);
    let n = ws.len();
    let mut ey = vec![0.0f64; nsamples * d];
    for i in 0..n_added {
        for j in 0..n {
            for k in 0..d {
                ey[i * d + k] = madd(ys[(i * n + j) * d + k], ws[j], ey[i * d + k]);
            }
        }
    }
    Ok(Array2::from_shape_vec((nsamples, d), ey).unwrap().into_pyarray(py))
}

/// (mask_aug, eyAdj_aug) for handing the selection to sklearn.
#[pyfunction]
fn kernel_row_aug<'py>(py: Python<'py>, ey_adj: PyReadonlyArray1<f64>, total: f64, masks: PyReadonlyArray2<f64>, kw: PyReadonlyArray1<f64>)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (ns, m) = (masks.shape()[0], masks.shape()[1]);
    let (xa, ya) = l1_aug(ey_adj.as_slice()?, total, masks.as_slice()?, kw.as_slice()?, ns, m);
    Ok((Array2::from_shape_vec((2 * ns, m), xa).unwrap().into_pyarray(py), Array1::from(ya).into_pyarray(py)))
}

/// l1 feature selection in Rust (code 1 num_features(param), 2 aic, 3 bic, 4 Lasso(alpha=param)).
/// Returns None when sklearn would warn, raise, or see non-finite input: the caller then runs sklearn.
#[pyfunction]
fn kernel_row_select(py: Python<'_>, ey_adj: PyReadonlyArray1<f64>, total: f64, masks: PyReadonlyArray2<f64>, kw: PyReadonlyArray1<f64>,
                     code: u32, param: f64) -> PyResult<Option<Vec<usize>>> {
    let (ns, m) = (masks.shape()[0], masks.shape()[1]);
    let (xa, ya) = l1_aug(ey_adj.as_slice()?, total, masks.as_slice()?, kw.as_slice()?, ns, m);
    if xa.iter().chain(ya.iter()).any(|v| !v.is_finite()) || ns == 0 {
        return Ok(None);
    }
    let l1 = L1::from_code(code, param);
    let (r, ev) = py.detach(|| l1_select(&xa, &ya, 2 * ns, m, l1));
    Ok(if ev { None } else { r.ok() })
}

/// Weighted least squares on the selected columns -> (phi over the M varying groups, singular).
#[pyfunction]
fn kernel_row_wls<'py>(py: Python<'py>, ey_adj: PyReadonlyArray1<f64>, total: f64, masks: PyReadonlyArray2<f64>, kw: PyReadonlyArray1<f64>,
                       nonzero: Vec<usize>) -> PyResult<(Bound<'py, PyArray1<f64>>, bool)> {
    let (ns, m) = (masks.shape()[0], masks.shape()[1]);
    let (phi, singular) = wls(ey_adj.as_slice()?, total, masks.as_slice()?, kw.as_slice()?, ns, m, &nonzero);
    Ok((Array1::from(phi).into_pyarray(py), singular))
}

/// Debug/test hook: support of make_pipeline(StandardScaler(with_mean=False), LassoLarsIC).fit(x, y).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn _lasso_lars_ic_support(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, bic: bool) -> PyResult<(Vec<usize>, Vec<Vec<f64>>, f64, Vec<f64>)> {
    let (n, p) = (x.shape()[0], x.shape()[1]);
    lasso::lasso_lars_ic_dbg(x.as_slice()?, y.as_slice()?, n, p, bic)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("too few samples"))
}

pub fn register_batch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_lasso_lars_ic_support, m)?)?;
    m.add_class::<KernelBatch>()?;
    m.add_function(wrap_pyfunction!(kernel_prepare_batch, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_prepare_batch_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_solve_batch, m)?)?;
    for f in [
        wrap_pyfunction!(kernel_row_varying, m)?,
        wrap_pyfunction!(kernel_row_varying_sparse, m)?,
        wrap_pyfunction!(kernel_row_plan, m)?,
        wrap_pyfunction!(kernel_row_synth, m)?,
        wrap_pyfunction!(kernel_row_synth_sparse, m)?,
        wrap_pyfunction!(kernel_row_ey, m)?,
        wrap_pyfunction!(kernel_row_aug, m)?,
        wrap_pyfunction!(kernel_row_select, m)?,
        wrap_pyfunction!(kernel_row_wls, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
