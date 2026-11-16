//! Feature selection used by KernelExplainer.solve for l1_reg="aic"/"bic"/float, and the singular
//! fallback of its weighted least squares:
//!   - make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion)).fit(X, y).coef_ != 0
//!     (sklearn's lars_path(method="lasso", Gram="auto") + information criterion + OLS noise variance)
//!   - Lasso(alpha).fit(X, y).coef_ != 0 (sklearn's cyclic coordinate descent with gap-safe screening)
//!   - np.linalg.lstsq (minimum-norm least squares, SVD with rcond = eps * max(n, p))
//! Matrices are row-major n x p. Only the support (and through it the final WLS) reaches shap's
//! output, so float rounding differences vs BLAS only matter at exact ties.

thread_local! {
    static SK_EVENT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// Marks that sklearn would warn (ConvergenceWarning / numpy RuntimeWarning) or raise at this point of the
/// ported algorithm. Callers that must reproduce sklearn's side effects then rerun sklearn itself.
pub fn sk_event() {
    SK_EVENT.with(|c| c.set(true));
}
/// Returns and clears the flag set by `sk_event`.
pub fn take_sk_event() -> bool {
    SK_EVENT.with(|c| c.replace(false))
}

/// numpy pairwise summation (contiguous float64 reduction), used where numpy reduces a contiguous axis.
pub fn np_sum(a: &[f64]) -> f64 {
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

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

/// Solve (L L^T) x = b in place, L lower-triangular with leading dimension ld (LAPACK potrs).
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

/// Reference BLAS drotg: returns (r, c, s).
fn rotg(a: f64, b: f64) -> (f64, f64, f64) {
    let roe = if a.abs() > b.abs() { a } else { b };
    let scale = a.abs() + b.abs();
    if scale == 0.0 {
        return (0.0, 1.0, 0.0);
    }
    let mut r = scale * ((a / scale).powi(2) + (b / scale).powi(2)).sqrt();
    if roe < 0.0 {
        r = -r;
    }
    (r, a / r, b / r)
}

/// sklearn.utils.arrayfuncs.cholesky_delete on L[:n, :n] (leading dimension ld).
fn cholesky_delete(l: &mut [f64], ld: usize, n: usize, go_out: usize) {
    for i in go_out..n - 1 {
        for k in 0..i + 2 {
            l[i * ld + k] = l[(i + 1) * ld + k];
        }
    }
    for i in go_out..n - 1 {
        let (r, mut c, mut s) = rotg(l[i * ld + i], l[i * ld + i + 1]);
        l[i * ld + i] = r;
        if l[i * ld + i] < 0.0 {
            l[i * ld + i] = l[i * ld + i].abs();
            c = -c;
            s = -s;
        }
        l[i * ld + i + 1] = 0.0;
        for row in i + 1..n {
            let x = l[row * ld + i];
            let y = l[row * ld + i + 1];
            l[row * ld + i] = c * x + s * y;
            l[row * ld + i + 1] = c * y - s * x;
        }
    }
}

/// sklearn _lars_path_solver with a precomputed Gram matrix (X.shape[0] > X.shape[1]), alpha_min=0,
/// return_path=True. Returns the coefficient path (one p-vector per step).
fn lars_path_gram(gram0: &[f64], cov0: &[f64], p: usize, n_samples: usize, max_iter: usize, lasso: bool) -> Vec<Vec<f64>> {
    let mut gram = gram0.to_vec();
    let mut cov_full = cov0.to_vec(); // Cov = cov_full[start..]; drops prepend
    let mut start = 0usize;
    let max_features = max_iter.min(p);
    let ld = max_features.max(1);
    let mut l = vec![0.0f64; ld * ld];
    let mut indices: Vec<usize> = (0..p).collect();
    let mut active: Vec<usize> = Vec::new();
    let mut sign_active = vec![0.0f64; ld];
    let mut coefs: Vec<Vec<f64>> = vec![vec![0.0; p]];
    let mut alphas: Vec<f64> = vec![0.0];
    let (mut n_iter, mut n_active) = (0usize, 0usize);
    let tiny32 = f32::MIN_POSITIVE as f64;
    let equality_tolerance = f32::EPSILON as f64;
    let eps = f64::EPSILON;
    let mut drop = false;

    loop {
        while coefs.len() <= n_iter {
            coefs.push(vec![0.0; p]);
            alphas.push(0.0);
        }
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
        alphas[n_iter] = c / n_samples as f64;
        if alphas[n_iter] <= equality_tolerance {
            if alphas[n_iter].abs() > equality_tolerance {
                // not reachable with alpha_min = 0 (alpha >= 0)
            } else if n_iter > 0 {
                let prev_a = alphas[n_iter - 1];
                if prev_a - alphas[n_iter] == 0.0 {
                    sk_event();
                }
                let ss = prev_a / (prev_a - alphas[n_iter]);
                let prev = coefs[n_iter - 1].clone();
                let cur = &mut coefs[n_iter];
                for j in 0..p {
                    cur[j] = prev[j] + ss * (cur[j] - prev[j]);
                }
            }
            alphas[n_iter] = 0.0;
            break;
        }
        if n_iter >= max_iter || n_active >= p {
            break;
        }
        if !drop {
            sign_active[n_active] = if c_ > 0.0 { 1.0 } else if c_ < 0.0 { -1.0 } else { 0.0 };
            let (m, nn) = (n_active, c_idx + n_active);
            cov_full.swap(start + c_idx, start);
            indices.swap(nn, m);
            start += 1;
            // Gram[m], Gram[n] = swap rows; then columns
            for k in 0..p {
                gram.swap(m * p + k, nn * p + k);
            }
            for k in 0..p {
                gram.swap(k * p + m, k * p + nn);
            }
            let c2 = gram[n_active * p + n_active];
            for k in 0..n_active {
                l[n_active * ld + k] = gram[n_active * p + k];
            }
            if n_active > 0 {
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
                sk_event(); // ConvergenceWarning: regressors in active set degenerate
                start -= 1;
                cov_full[start] = 0.0;
                cov_full.swap(start + c_idx, start);
                continue;
            }
            active.push(indices[n_active]);
            n_active += 1;
        }
        if lasso && n_iter > 0 && alphas[n_iter - 1] < alphas[n_iter] {
            sk_event(); // ConvergenceWarning: early stopping the lars path
            break;
        }
        let na = n_active;
        let mut ls: Vec<f64> = sign_active[..na].to_vec();
        chol_solve(&l, ld, na, &mut ls);
        let aa;
        if na == 1 && ls[0] == 0.0 {
            ls[0] = 1.0;
            aa = 1.0;
        } else {
            let mut a = 1.0 / ls.iter().zip(&sign_active[..na]).map(|(v, s)| v * s).sum::<f64>().sqrt();
            if !a.is_finite() {
                sk_event();
                let mut i = 0;
                let mut l2 = vec![0.0f64; na * na];
                for r in 0..na {
                    for cc in 0..na {
                        l2[r * na + cc] = l[r * ld + cc];
                    }
                }
                while !a.is_finite() {
                    for d in 0..na {
                        l2[d * na + d] += 2f64.powi(i) * eps;
                    }
                    ls = sign_active[..na].to_vec();
                    chol_solve(&l2, na, na, &mut ls);
                    let tmp = ls.iter().zip(&sign_active[..na]).map(|(v, s)| v * s).sum::<f64>().max(eps);
                    a = 1.0 / tmp.sqrt();
                    i += 1;
                }
            }
            aa = a;
            for v in ls.iter_mut() {
                *v *= aa;
            }
        }
        // corr_eq_dir = Gram[:na, na:].T @ ls
        let corr: Vec<f64> = (na..p)
            .map(|j| np_around15((0..na).map(|k| gram[k * p + j] * ls[k]).sum::<f64>()))
            .collect();
        let cov = &cov_full[start..];
        if corr.iter().any(|cd| aa - cd + tiny32 == 0.0 || aa + cd + tiny32 == 0.0) || ls.iter().any(|l_| l_ + tiny32 == 0.0) {
            sk_event(); // numpy divide-by-zero RuntimeWarning
        }
        let g1 = min_pos(cov.iter().zip(&corr).map(|(cv, cd)| (c - cv) / (aa - cd + tiny32)));
        let g2 = min_pos(cov.iter().zip(&corr).map(|(cv, cd)| (c + cv) / (aa + cd + tiny32)));
        let mut gamma = g1.min(g2).min(c / aa);
        drop = false;
        let cur_coef = &coefs[n_iter];
        let z: Vec<f64> = active.iter().zip(&ls).map(|(&a, l_)| -cur_coef[a] / (l_ + tiny32)).collect();
        let z_pos = min_pos(z.iter().copied());
        let mut idx: Vec<usize> = Vec::new();
        if z_pos < gamma {
            idx = (0..z.len()).rev().filter(|&i| z[i] == z_pos).collect();
            for &i in &idx {
                sign_active[i] = -sign_active[i];
            }
            if lasso {
                gamma = z_pos;
            }
            drop = true;
        }
        n_iter += 1;
        while coefs.len() <= n_iter {
            coefs.push(vec![0.0; p]);
            alphas.push(0.0);
        }
        let prev = coefs[n_iter - 1].clone();
        let cur = &mut coefs[n_iter];
        for (k, &a) in active.iter().enumerate() {
            cur[a] = prev[a] + gamma * ls[k];
        }
        for (cv, cd) in cov_full[start..].iter_mut().zip(&corr) {
            *cv -= gamma * cd;
        }
        if drop && lasso {
            for &ii in &idx {
                cholesky_delete(&mut l, ld, n_active, ii);
            }
            n_active -= 1;
            let drop_idx: Vec<usize> = idx.iter().map(|&ii| active.remove(ii)).collect();
            for &ii in &idx {
                for i in ii..n_active {
                    indices.swap(i, i + 1);
                    for k in 0..p {
                        gram.swap(i * p + k, (i + 1) * p + k);
                    }
                    for k in 0..p {
                        gram.swap(k * p + i, k * p + i + 1);
                    }
                }
            }
            let cur = &coefs[n_iter];
            let temp: Vec<f64> = drop_idx.iter().map(|&d| cov0[d] - dot(&gram0[d * p..(d + 1) * p], cur)).collect();
            // Cov = np.r_[temp, Cov]
            let mut nc = temp;
            nc.extend_from_slice(&cov_full[start..]);
            cov_full = nc;
            start = 0;
            // sign_active = np.delete(sign_active, idx); append 0
            let mut sa: Vec<f64> = sign_active.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, &v)| v).collect();
            while sa.len() < ld {
                sa.push(0.0);
            }
            sign_active = sa;
        }
    }
    coefs.truncate(n_iter + 1);
    coefs
}

/// Column means over rows (numpy reduction along axis 0 of a C-ordered array: sequential per column).
fn col_means(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            s[j] += x[i * p + j];
        }
    }
    s.iter().map(|v| v / n as f64).collect()
}

pub enum IcError {
    TooFewSamples,
}

/// make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion)).fit(X, y); returns the
/// indices of the nonzero coefficients.
pub fn lasso_lars_ic(x: &[f64], y: &[f64], n: usize, p: usize, bic: bool) -> Result<Vec<usize>, IcError> {
    lasso_lars_ic_dbg(x, y, n, p, bic).map(|r| r.0)
}

#[allow(clippy::type_complexity)]
pub fn lasso_lars_ic_dbg(x: &[f64], y: &[f64], n: usize, p: usize, bic: bool) -> Result<(Vec<usize>, Vec<Vec<f64>>, f64, Vec<f64>), IcError> {
    // StandardScaler(with_mean=False): scale by the population std (two-pass), constant features -> 1
    let mean = col_means(x, n, p);
    let mut corr = vec![0.0f64; p];
    let mut sq = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            let t = x[i * p + j] - mean[j];
            corr[j] += t;
            sq[j] += t * t;
        }
    }
    let scale: Vec<f64> = (0..p)
        .map(|j| {
            let var = (sq[j] - corr[j] * corr[j] / n as f64) / n as f64;
            let eps = f64::EPSILON;
            let upper = n as f64 * eps * var + (n as f64 * mean[j] * eps).powi(2);
            if var <= upper {
                1.0
            } else {
                var.sqrt()
            }
        })
        .collect();
    let mut xs: Vec<f64> = (0..n * p).map(|k| x[k] / scale[k % p]).collect();
    // LassoLarsIC: center X and y
    let xm = col_means(&xs, n, p);
    for i in 0..n {
        for j in 0..p {
            xs[i * p + j] -= xm[j];
        }
    }
    let ym = np_sum(y) / n as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - ym).collect();
    // lars_path(Gram="auto"): Gram = X^T X when n > p
    if n <= p + 1 {
        // _estimate_noise_variance raises ValueError (after the path); nothing else observable
        return Err(IcError::TooFewSamples);
    }
    let mut gram = vec![0.0f64; p * p];
    let mut cov = vec![0.0f64; p];
    for i in 0..n {
        let r = &xs[i * p..(i + 1) * p];
        for a in 0..p {
            let ra = r[a];
            if ra != 0.0 {
                for b in 0..p {
                    gram[a * p + b] += ra * r[b];
                }
            }
            cov[a] = ra.mul_add(yc[i], cov[a]); // numpy's gemv (X.T @ y) accumulates with FMA in row order
        }
    }
    let path = lars_path_gram(&gram, &cov, p, n, 500, true);
    let factor = if bic { (n as f64).ln() } else { 2.0 };
    // LinearRegression(fit_intercept=False).fit(X, y).predict(X): explicit residual of the lstsq solution
    let beta = lstsq(&xs, &yc, n, p);
    let rss_ols: f64 = (0..n).map(|i| (yc[i] - dot(&xs[i * p..(i + 1) * p], &beta)).powi(2)).sum();
    let mut sigma2 = rss_ols / (n - p - 1) as f64;
    if sigma2 <= 0.0 || !sigma2.is_finite() {
        sk_event();
        // exact OLS fit: sklearn's gelsd leaves a rounding-level residual (~1e-30) where ours can be exactly
        // 0 (log(0) -> NaN criteria). Any tiny positive variance selects the same exact-fit step.
        sigma2 = f64::MIN_POSITIVE;
    }
    let mut best = 0usize;
    let mut best_c = f64::INFINITY;
    let mut crits = vec![];
    for (k, coef) in path.iter().enumerate() {
        let mut rss = 0.0;
        for i in 0..n {
            let r = yc[i] - dot(&xs[i * p..(i + 1) * p], coef);
            rss += r * r;
        }
        let df = coef.iter().filter(|c| c.abs() > f64::EPSILON).count() as f64;
        let crit = n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln() + rss / sigma2 + factor * df;
        if !crit.is_finite() {
            sk_event();
        }
        crits.push(crit);
        // np.argmin: first minimum (NaN wins in numpy; treat NaN as minimum too)
        if crit.is_nan() {
            if !best_c.is_nan() {
                best = k;
                best_c = crit;
            }
        } else if !best_c.is_nan() && crit < best_c {
            best = k;
            best_c = crit;
        }
    }
    Ok(((0..p).filter(|&j| path[best][j] != 0.0).collect(), path, sigma2, crits))
}

/// Lasso(alpha).fit(X, y): cyclic coordinate descent (sklearn cd_fast.enet_coordinate_descent,
/// beta = 0, gap-safe screening, tol=1e-4, max_iter=1000). Returns the nonzero coefficients, or None
/// when they are not finite (sklearn raises ValueError).
pub fn lasso_cd(x: &[f64], y: &[f64], n: usize, p: usize, alpha: f64) -> Option<Vec<usize>> {
    // center (fit_intercept=True); X is Fortran-ordered in sklearn, so column means are contiguous sums
    let cols: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let c: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
            let m = np_sum(&c) / n as f64;
            c.iter().map(|v| v - m).collect()
        })
        .collect();
    let ym = np_sum(y) / n as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - ym).collect();
    let alpha = alpha * n as f64;
    let max_iter = 1000;
    let norm2: Vec<f64> = cols.iter().map(|c| dot(c, c)).collect();
    let mut w = vec![0.0f64; p];
    let mut r = yc.clone();
    let tol = 1e-4 * dot(&yc, &yc);
    let mut xta = vec![0.0f64; p];
    let gap_fn = |w: &[f64], r: &[f64], xta: &mut [f64]| -> (f64, f64) {
        let r_norm2 = dot(r, r);
        let ry = dot(r, &yc);
        if alpha == 0.0 {
            for j in 0..p {
                xta[j] = dot(&cols[j], r);
            }
            let d = dot(xta, xta);
            return (d, d);
        }
        for j in 0..p {
            xta[j] = dot(&cols[j], r);
        }
        let dual_norm = xta.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let l1: f64 = w.iter().map(|v| v.abs()).sum();
        let primal = 0.5 * r_norm2 + alpha * l1;
        let scale = if dual_norm > alpha { alpha / dual_norm } else { 1.0 };
        let dual = -0.5 * scale * scale * r_norm2 + scale * ry;
        (primal - dual, dual_norm)
    };
    let do_screening = alpha != 0.0;
    let (mut gap, mut dual_norm) = gap_fn(&w, &r, &mut xta);
    if gap <= tol {
        return Some(vec![]);
    }
    let mut active: Vec<usize> = (0..p).collect();
    let mut excluded = vec![false; p];
    let screen = |w: &mut [f64], r: &mut [f64], xta: &[f64], gap: f64, dual_norm: f64, active: &mut Vec<usize>, excluded: &mut [bool], first: bool| {
        active.clear();
        for j in 0..p {
            if first {
                if norm2[j] == 0.0 {
                    w[j] = 0.0;
                    excluded[j] = true;
                    continue;
                }
            } else if excluded[j] {
                continue;
            }
            let xj_theta = xta[j] / alpha.max(dual_norm);
            let d_j = (1.0 - xj_theta.abs()) / norm2[j].sqrt();
            if d_j <= (2.0 * gap).sqrt() / alpha {
                active.push(j);
                excluded[j] = false;
            } else {
                if w[j] != 0.0 {
                    for i in 0..n {
                        r[i] += w[j] * cols[j][i];
                    }
                    w[j] = 0.0;
                }
                excluded[j] = true;
            }
        }
    };
    if do_screening {
        screen(&mut w, &mut r, &xta, gap, dual_norm, &mut active, &mut excluded, true);
    }
    let mut converged = false;
    for n_iter in 0..max_iter {
        let mut w_max = 0.0f64;
        let mut d_w_max = 0.0f64;
        for &j in active.iter() {
            if norm2[j] == 0.0 {
                continue;
            }
            let w_j = w[j];
            let tmp = dot(&cols[j], &r) + w_j * norm2[j];
            let s = if tmp > 0.0 { 1.0 } else if tmp < 0.0 { -1.0 } else { 0.0 };
            w[j] = s * (tmp.abs() - alpha).max(0.0) / norm2[j];
            if w[j] != w_j {
                let d = w_j - w[j];
                for i in 0..n {
                    r[i] += d * cols[j][i];
                }
            }
            d_w_max = d_w_max.max((w[j] - w_j).abs());
            w_max = w_max.max(w[j].abs());
        }
        if w_max == 0.0 || d_w_max / w_max <= 1e-4 || n_iter == max_iter - 1 {
            let g = gap_fn(&w, &r, &mut xta);
            gap = g.0;
            dual_norm = g.1;
            if gap <= tol {
                converged = true;
                break;
            }
            if do_screening {
                screen(&mut w, &mut r, &xta, gap, dual_norm, &mut active, &mut excluded, false);
            }
        }
    }
    if !converged {
        sk_event(); // ConvergenceWarning: objective did not converge
    }
    if w.iter().any(|v| !v.is_finite()) {
        sk_event();
        return None;
    }
    Some((0..p).filter(|&j| w[j] != 0.0).collect())
}

/// np.linalg.lstsq(A, b, rcond=None)[0]: minimum-norm least squares via one-sided Jacobi SVD.
pub fn lstsq(a: &[f64], b: &[f64], n: usize, k: usize) -> Vec<f64> {
    if k == 0 {
        return vec![];
    }
    // columns of U*S (n x k, column-major) and V (k x k)
    let mut u: Vec<Vec<f64>> = (0..k).map(|j| (0..n).map(|i| a[i * k + j]).collect()).collect();
    let mut v: Vec<Vec<f64>> = (0..k).map(|j| (0..k).map(|i| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..k {
            for q in p + 1..k {
                let alpha = dot(&u[p], &u[p]);
                let beta = dot(&u[q], &u[q]);
                let gamma = dot(&u[p], &u[q]);
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt());
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..n {
                    let (x, y) = (u[p][i], u[q][i]);
                    u[p][i] = c * x - s * y;
                    u[q][i] = s * x + c * y;
                }
                for i in 0..k {
                    let (x, y) = (v[p][i], v[q][i]);
                    v[p][i] = c * x - s * y;
                    v[q][i] = s * x + c * y;
                }
            }
        }
        if off <= 1e-15 {
            break;
        }
    }
    let sig: Vec<f64> = u.iter().map(|c| dot(c, c).sqrt()).collect();
    let smax = sig.iter().cloned().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * (n.max(k) as f64) * smax;
    let mut x = vec![0.0f64; k];
    for j in 0..k {
        if sig[j] > cutoff {
            let coef = dot(&u[j], b) / (sig[j] * sig[j]);
            for i in 0..k {
                x[i] += coef * v[j][i];
            }
        }
    }
    x
}
