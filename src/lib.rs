use numpy::{IntoPyArray, PyUntypedArrayMethods, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

mod fast;
mod fast_indep;
mod fast_indep_tf;
mod fast_inter;
mod indep;
mod kernel;
mod lasso;
mod tree;

use tree::{Ensemble, Tree};

fn vec_of<T: Copy + numpy::Element>(a: &PyReadonlyArrayDyn<T>) -> Vec<T> {
    a.as_array().iter().copied().collect()
}

/// compute_expectations(children_left, children_right, node_sample_weight, values) -> max_depth.
/// `values` (n_nodes x n_outputs, float64, C-contiguous) is updated in place.
#[pyfunction]
fn compute_expectations(
    cl: PyReadonlyArray1<i32>,
    cr: PyReadonlyArray1<i32>,
    nsw: PyReadonlyArray1<f64>,
    mut values: PyReadwriteArrayDyn<f64>,
) -> PyResult<usize> {
    let nout = if values.ndim() > 1 { values.shape()[1] } else { 1 };
    let cl = cl.as_slice()?;
    let cr = cr.as_slice()?;
    let nsw = nsw.as_slice()?;
    let v = values.as_slice_mut()?;
    Ok(tree::compute_expectations(cl, cr, nsw, v, nout, 0, 0))
}

/// dense_tree_update_weights for a single tree: node_sample_weight[i] += #rows of X passing node i.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dense_tree_update_weights(
    cl: PyReadonlyArray1<i32>,
    cr: PyReadonlyArray1<i32>,
    cd: PyReadonlyArray1<i32>,
    feat: PyReadonlyArray1<i32>,
    thr: PyReadonlyArray1<f64>,
    ttype: PyReadonlyArray1<i32>,
    mut nsw: PyReadwriteArray1<f64>,
    x: PyReadonlyArray2<f64>,
    xm: PyReadonlyArray2<bool>,
) -> PyResult<()> {
    let m = x.shape()[1];
    let t = Tree {
        cl: cl.as_slice()?,
        cr: cr.as_slice()?,
        cd: cd.as_slice()?,
        feat: feat.as_slice()?,
        thr: thr.as_slice()?,
        ttype: ttype.as_slice()?,
        values: &[],
        nsw: &[],
        nout: 0,
    };
    let xs = x.as_slice()?;
    let xms = xm.as_slice()?;
    let w = nsw.as_slice_mut()?;
    for i in 0..x.shape()[0] {
        let (xr, xmr) = (&xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m]);
        let mut node = 0usize;
        loop {
            w[node] += 1.0;
            if t.cl[node] < 0 {
                break;
            }
            node = t.next(node, xr, xmr);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn make_ensemble(
    cl: &PyReadonlyArrayDyn<i32>,
    cr: &PyReadonlyArrayDyn<i32>,
    cd: &PyReadonlyArrayDyn<i32>,
    feat: &PyReadonlyArrayDyn<i32>,
    thr: &PyReadonlyArrayDyn<f64>,
    ttype: &PyReadonlyArrayDyn<i32>,
    values: &PyReadonlyArrayDyn<f64>,
    nsw: Option<&PyReadonlyArrayDyn<f64>>,
    max_depth: usize,
    tree_limit: usize,
    base_offset: &PyReadonlyArrayDyn<f64>,
) -> Ensemble {
    let vs = values.shape();
    let max_nodes = vs[1];
    let nout = vs[2];
    Ensemble {
        cl: vec_of(cl),
        cr: vec_of(cr),
        cd: vec_of(cd),
        feat: vec_of(feat),
        thr: vec_of(thr),
        ttype: vec_of(ttype),
        values: vec_of(values),
        nsw: nsw.map(vec_of).unwrap_or_default(),
        max_depth,
        tree_limit,
        base_offset: vec_of(base_offset),
        max_nodes,
        nout,
    }
}

#[pyfunction]
#[pyo3(signature = (cl, cr, cd, feat, thr, ttype, values, max_depth, tree_limit, base_offset, transform, x, xm, y=None))]
#[allow(clippy::too_many_arguments)]
fn dense_tree_predict<'py>(
    py: Python<'py>,
    cl: PyReadonlyArrayDyn<i32>,
    cr: PyReadonlyArrayDyn<i32>,
    cd: PyReadonlyArrayDyn<i32>,
    feat: PyReadonlyArrayDyn<i32>,
    thr: PyReadonlyArrayDyn<f64>,
    ttype: PyReadonlyArrayDyn<i32>,
    values: PyReadonlyArrayDyn<f64>,
    max_depth: usize,
    tree_limit: usize,
    base_offset: PyReadonlyArrayDyn<f64>,
    transform: u32,
    x: PyReadonlyArray2<f64>,
    xm: PyReadonlyArray2<bool>,
    y: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let e = make_ensemble(&cl, &cr, &cd, &feat, &thr, &ttype, &values, None, max_depth, tree_limit, &base_offset);
    let (n, m) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice()?.to_vec();
    let xms = xm.as_slice()?.to_vec();
    let ys: Option<Vec<f64>> = y.map(|a| a.as_array().to_vec());
    let tf = tree::get_transform(transform);
    let nout = e.nout;
    let out = py.detach(|| {
        let mut out = vec![0.0f64; n * nout];
        out.par_chunks_mut(nout.max(1)).enumerate().for_each(|(i, o)| {
            let yi = ys.as_ref().map(|v| v[i]).unwrap_or(0.0);
            tree::predict_row(&e, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], yi, tf, o);
        });
        out
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((n, nout), out).unwrap().into_pyarray(py))
}

/// dense_tree_shap: returns phi of shape (N, M+1, nout) or (N, M+1, M+1, nout) for interactions.
#[pyfunction]
#[pyo3(signature = (cl, cr, cd, feat, thr, ttype, values, nsw, max_depth, x, xm, y, r, rm, tree_limit, base_offset, feature_dependence, transform, interactions))]
#[allow(clippy::too_many_arguments)]
fn dense_tree_shap<'py>(
    py: Python<'py>,
    cl: PyReadonlyArrayDyn<i32>,
    cr: PyReadonlyArrayDyn<i32>,
    cd: PyReadonlyArrayDyn<i32>,
    feat: PyReadonlyArrayDyn<i32>,
    thr: PyReadonlyArrayDyn<f64>,
    ttype: PyReadonlyArrayDyn<i32>,
    values: PyReadonlyArrayDyn<f64>,
    nsw: PyReadonlyArrayDyn<f64>,
    max_depth: usize,
    x: PyReadonlyArray2<f64>,
    xm: PyReadonlyArray2<bool>,
    y: Option<PyReadonlyArray1<f64>>,
    r: Option<PyReadonlyArray2<f64>>,
    rm: Option<PyReadonlyArray2<bool>>,
    tree_limit: usize,
    base_offset: PyReadonlyArrayDyn<f64>,
    feature_dependence: u32,
    transform: u32,
    interactions: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let e = make_ensemble(&cl, &cr, &cd, &feat, &thr, &ttype, &values, Some(&nsw), max_depth, tree_limit, &base_offset);
    let (n, m) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice()?.to_vec();
    let xms = xm.as_slice()?.to_vec();
    let ys: Option<Vec<f64>> = y.map(|a| a.as_array().to_vec());
    let rs: Option<Vec<f64>> = r.map(|a| a.as_array().iter().copied().collect());
    let rms: Option<Vec<bool>> = rm.map(|a| a.as_array().iter().copied().collect());
    let tf = tree::get_transform(transform);
    let nout = e.nout;

    if feature_dependence == 0 {
        if interactions {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "FEATURE_DEPENDENCE::independent does not support interactions!",
            ));
        }
        let rs = rs.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("background data required"))?;
        let rms = rms.unwrap();
        let num_r = rs.len() / m.max(1);
        let row = (m + 1) * nout;
        let out = py.detach(|| independent(&e, &xs, &xms, ys.as_deref(), &rs, &rms, n, m, num_r, tf));
        let _ = row;
        return Ok(numpy::ndarray::Array3::from_shape_vec((n, m + 1, nout), out).unwrap().into_dyn().into_pyarray(py));
    }

    if !interactions {
        let out = py.detach(|| {
            if std::env::var("SHAP_RS_NO_FAST").is_ok() {
                let row = (m + 1) * nout;
                let mut out = vec![0.0f64; n * row];
                out.par_chunks_mut(row).enumerate().for_each_init(
                    || tree::path_buf(e.max_depth),
                    |buf, (i, o)| {
                        tree::path_dependent_row(&e, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], m, o, buf);
                    },
                );
                out
            } else {
                fast::path_dependent(&e, &xs, &xms, n, m)
            }
        });
        return Ok(numpy::ndarray::Array3::from_shape_vec((n, m + 1, nout), out).unwrap().into_dyn().into_pyarray(py));
    }

    let crs = (m + 1) * nout;
    let row = (m + 1) * crs;
    let infos: Option<Vec<fast_inter::TreeInfo>> = if std::env::var("SHAP_RS_NO_FAST").is_ok() {
        None
    } else {
        (0..e.tree_limit).into_par_iter().map(|k| fast_inter::build(&e.tree(k))).collect()
    };
    let out = py.detach(|| {
        let mut out = vec![0.0f64; n * row];
        if let Some(infos) = &infos {
            let threads = rayon::current_num_threads();
            if e.tree_limit >= 2 * threads && n * row * threads < (1usize << 27) {
                return fast_inter::interactions_memo(&e, infos, &xs, &xms, n, m);
            }
            out.par_chunks_mut(row).enumerate().for_each_init(
                || (fast_inter::Scratch::new(e.max_nodes, e.max_depth.max(32)), vec![0.0; crs]),
                |(s, shap), (i, o)| {
                    fast_inter::row(&e, infos, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], m, o, shap, s);
                },
            );
            return out;
        }
        let (uf, amount) = tree::unique_features(&e, m);
        out.par_chunks_mut(row).enumerate().for_each_init(
            || (tree::path_buf(e.max_depth), vec![0.0; crs], vec![0.0; crs], vec![0.0; crs]),
            |(buf, diag, on, off), (i, o)| {
                tree::interactions_row(
                    &e, &uf, amount, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], m, o, buf, diag, on, off,
                );
            },
        );
        out
    });
    Ok(numpy::ndarray::Array4::from_shape_vec((n, m + 1, m + 1, nout), out).unwrap().into_dyn().into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
fn independent(
    e: &Ensemble,
    xs: &[f64],
    xms: &[bool],
    ys: Option<&[f64]>,
    rs: &[f64],
    rms: &[bool],
    n: usize,
    m: usize,
    num_r: usize,
    tf: Option<tree::Transform>,
) -> Vec<f64> {
    let nout = e.nout;
    let row = (m + 1) * nout;
    let mw = indep::memoized_weights(e.max_depth);
    let mut out = vec![0.0f64; n * row];
    for oind in 0..nout {
        let nodes = indep::build_nodes(e, oind);
        if tf.is_none()
            && std::env::var("SHAP_RS_NO_FAST").is_err()
            && fast_indep::independent_identity(e, &nodes, &mw, oind, xs, xms, rs, rms, n, m, num_r, &mut out)
        {
            continue;
        }
        // margins of each reference (only needed with a transform)
        let margins_r: Vec<f64> = if tf.is_some() {
            (0..num_r)
                .into_par_iter()
                .map(|j| margin(e, &rs[j * m..(j + 1) * m], &rms[j * m..(j + 1) * m], oind))
                .collect()
        } else {
            vec![]
        };
        if let Some(t) = tf {
            if std::env::var("SHAP_RS_NO_FAST").is_err() {
                let margins_x: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| margin(e, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], oind))
                    .collect();
                if fast_indep_tf::independent_transform(
                    e, &nodes, &mw, oind, xs, xms, ys, rs, rms, n, m, num_r, &margins_x, &margins_r, t, &mut out,
                ) {
                    continue;
                }
            }
        }
        // directions of every reference at every node, [tree][ref][node]
        let mn = e.max_nodes;
        let mut rdirs = vec![3u8; e.tree_limit * num_r * mn];
        rdirs.par_chunks_mut(num_r * mn).enumerate().for_each(|(k, chunk)| {
            let tree = &nodes[k * mn..(k + 1) * mn];
            for j in 0..num_r {
                indep::directions(tree, &rs[j * m..(j + 1) * m], &rms[j * m..(j + 1) * m], &mut chunk[j * mn..(j + 1) * mn]);
            }
        });
        out.par_chunks_mut(row).enumerate().for_each_init(
            || (indep::Scratch::new(e.max_nodes, e.max_depth, m), vec![0.0f64; num_r * (m + 1)], vec![3u8; mn]),
            |(s, tmps, xdir), (i, o)| {
                let x = &xs[i * m..(i + 1) * m];
                let xm = &xms[i * m..(i + 1) * m];
                let yi = ys.map(|v| v[i]).unwrap_or(0.0);
                let mx = if tf.is_some() { margin(e, x, xm, oind) } else { 0.0 };
                indep::independent_row(e, &nodes, &mw, oind, x, xm, yi, &rdirs, num_r, m, mx, &margins_r, tf, o, tmps, xdir, s);
            },
        );
    }
    out
}

fn margin(e: &Ensemble, x: &[f64], xm: &[bool], oind: usize) -> f64 {
    let mut v = e.base_offset[oind];
    for k in 0..e.tree_limit {
        let t = e.tree(k);
        v += t.values[t.leaf(x, xm) * e.nout + oind];
    }
    v
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dense_tree_saabas<'py>(
    py: Python<'py>,
    cl: PyReadonlyArrayDyn<i32>,
    cr: PyReadonlyArrayDyn<i32>,
    cd: PyReadonlyArrayDyn<i32>,
    feat: PyReadonlyArrayDyn<i32>,
    thr: PyReadonlyArrayDyn<f64>,
    ttype: PyReadonlyArrayDyn<i32>,
    values: PyReadonlyArrayDyn<f64>,
    tree_limit: usize,
    base_offset: PyReadonlyArrayDyn<f64>,
    x: PyReadonlyArray2<f64>,
    xm: PyReadonlyArray2<bool>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let e = make_ensemble(&cl, &cr, &cd, &feat, &thr, &ttype, &values, None, 0, tree_limit, &base_offset);
    let (n, m) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice()?.to_vec();
    let xms = xm.as_slice()?.to_vec();
    let nout = e.nout;
    let row = (m + 1) * nout;
    let out = py.detach(|| {
        let mut out = vec![0.0f64; n * row];
        out.par_chunks_mut(row).enumerate().for_each(|(i, o)| {
            tree::saabas_row(&e, &xs[i * m..(i + 1) * m], &xms[i * m..(i + 1) * m], m, o);
        });
        out
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((n, m + 1, nout), out).unwrap().into_dyn().into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_expectations, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_update_weights, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_predict, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_shap, m)?)?;
    m.add_function(wrap_pyfunction!(dense_tree_saabas, m)?)?;
    kernel::register_batch(m)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused(_: Option<Bound<'_, PyArray1<f64>>>) {}
