//! Interventional ("independent") Tree SHAP: faithful port of tree_shap_indep / dense_independent
//! from shap/cext/tree_shap.h, including its float32 node values, thresholds and weights.

use crate::tree::{Ensemble, Transform};

const FROM_NEITHER: u8 = 0;
const FROM_X_NOT_R: u8 = 1;
const FROM_R_NOT_X: u8 = 2;

#[derive(Clone, Copy, Default)]
pub struct Node {
    pub cl: i32,
    pub cr: i32,
    pub cd: i32,
    pub pnode: i32,
    pub feat: i64,
    pub pfeat: i64,
    pub thres: f32,
    pub value: f32,
}

fn bin_coeff(n: i32, k: i32) -> i32 {
    let mut res: i32 = 1;
    let mut k = k;
    if k > n.wrapping_sub(k) {
        k = n.wrapping_sub(k);
    }
    let mut i = 0;
    while i < k {
        res = res.wrapping_mul(n.wrapping_sub(i));
        res /= i + 1;
        i += 1;
    }
    res
}

/// memoized_weights exactly as dense_independent computes them (note the max_depth stride).
pub fn memoized_weights(max_depth: usize) -> Vec<f32> {
    let d = max_depth as u32;
    let mut w = vec![0f32; (max_depth + 1) * (max_depth + 1)];
    for n in 0..=d {
        for m in 0..=d {
            let bc = bin_coeff(n.wrapping_sub(1) as i32, m as i32);
            let denom = n.wrapping_mul(bc as u32);
            w[(n + d * m) as usize] = (1.0f64 / denom as f64) as f32;
        }
    }
    w
}

/// Reformatted trees (one Node array per tree, max_nodes each) for a given output index.
pub fn build_nodes(e: &Ensemble, oind: usize) -> Vec<Node> {
    let mn = e.max_nodes;
    let mut nodes = vec![Node::default(); e.tree_limit * mn];
    for i in 0..e.tree_limit {
        let nt = &mut nodes[i * mn..(i + 1) * mn];
        for j in 0..mn {
            let en = i * mn + j;
            nt[j].cl = e.cl[en];
            nt[j].cr = e.cr[en];
            nt[j].cd = e.cd[en];
            if j == 0 {
                nt[j].pnode = 0;
            }
            if e.cl[en] >= 0 {
                let c = e.cl[en] as usize;
                nt[c].pnode = j as i32;
                nt[c].pfeat = e.feat[en] as i64;
            }
            if e.cr[en] >= 0 {
                let c = e.cr[en] as usize;
                nt[c].pnode = j as i32;
                nt[c].pfeat = e.feat[en] as i64;
            }
            nt[j].thres = e.thr[en] as f32;
            nt[j].feat = e.feat[en] as i64;
            nt[j].value = e.values[en * e.nout + oind] as f32;
        }
    }
    nodes
}

/// Per-thread scratch space.
pub struct Scratch {
    pub pos: Vec<f32>,
    pub neg: Vec<f32>,
    pub flag: Vec<u8>,
    pub stack: Vec<i32>,
    pub hist: Vec<i16>,
}

impl Scratch {
    pub fn new(max_nodes: usize, max_depth: usize, m: usize) -> Self {
        Scratch {
            pos: vec![0.0; max_nodes],
            neg: vec![0.0; max_nodes],
            flag: vec![0; max_nodes],
            stack: vec![0; max_depth + 1],
            hist: vec![0; m],
        }
    }
}

#[inline(always)]
fn pick(d: u8, cl: i32, cr: i32, cd: i32) -> i32 {
    match d {
        0 => cl,
        1 => cr,
        2 => cd,
        _ => -1,
    }
}

/// Split direction of a row at every node of one tree, as tree_shap_indep decides it:
/// 2 = missing (default child), 1 = right (x > float32 threshold), 0 = left, 3 = undecidable/leaf.
pub fn directions(tree: &[Node], x: &[f64], xm: &[bool], out: &mut [u8]) {
    for (j, nd) in tree.iter().enumerate() {
        out[j] = if nd.cl < 0 || nd.feat < 0 {
            3
        } else {
            let f = nd.feat as usize;
            let t = nd.thres as f64;
            if xm[f] {
                2
            } else if x[f] > t {
                1
            } else if x[f] <= t {
                0
            } else {
                3
            }
        };
    }
}

/// tree_shap_indep for one tree, one x, one reference r. Adds into out (M+1 entries).
#[allow(clippy::too_many_arguments)]
pub fn tree_shap_indep(
    max_depth: usize,
    num_feats: usize,
    xdir: &[u8],
    rdir: &[u8],
    out: &mut [f64],
    mw: &[f32],
    tree: &[Node],
    s: &mut Scratch,
) {
    let pos = &mut s.pos;
    let neg = &mut s.neg;
    let flag = &mut s.flag;
    let stack = &mut s.stack;
    let hist = &mut s.hist;
    let md = max_depth as u32;

    let mut ns: usize = 0;
    hist.fill(0);
    let mut node: i32 = 0;
    let mut next_xnode: i32 = -1;
    let mut next_rnode: i32 = -1;
    let mut next_node: i32 = -1;
    let mut from_child: i32 = -1;
    let mut m_: u32 = 0;
    let mut n_: u32 = 0;

    let cur = tree[0];
    let feat = cur.feat as usize;
    let (cl, cr, cd) = (cur.cl, cur.cr, cur.cd);

    if cl < 0 {
        out[num_feats] += cur.value as f64;
        return;
    }

    next_xnode = pick(xdir[0], cl, cr, cd);
    next_rnode = pick(rdir[0], cl, cr, cd);

    if next_xnode != next_rnode {
        flag[next_xnode as usize] = FROM_X_NOT_R;
        flag[next_rnode as usize] = FROM_R_NOT_X;
    } else {
        flag[next_xnode as usize] = FROM_NEITHER;
    }

    if next_xnode == next_rnode {
        next_node = next_xnode;
    }
    if next_node < 0 {
        next_node = cl;
        if next_rnode == next_node {
            n_ += 1;
            hist[feat] -= 1;
        } else if next_xnode == next_node {
            m_ += 1;
            n_ += 1;
            hist[feat] += 1;
        }
    }
    stack[ns] = node;
    ns += 1;

    macro_rules! unwind {
        ($pfeat:expr, $from_flag:expr) => {{
            let pf = $pfeat as usize;
            if hist[pf] > 0 {
                hist[pf] -= 1;
            } else if hist[pf] < 0 {
                hist[pf] += 1;
            }
            if hist[pf] == 0 {
                if $from_flag == FROM_X_NOT_R {
                    n_ = n_.wrapping_sub(1);
                    m_ = m_.wrapping_sub(1);
                } else if $from_flag == FROM_R_NOT_X {
                    n_ = n_.wrapping_sub(1);
                }
            }
        }};
    }

    loop {
        node = next_node;
        let nu = node as usize;
        let cur = tree[nu];
        let cl = cur.cl;
        let cr = cur.cr;
        let cd = cur.cd;
        let pnode = cur.pnode;
        let pfeat = cur.pfeat;
        let from_flag = flag[nu];

        if cl < 0 {
            if m_ == 0 {
                out[num_feats] += cur.value as f64;
            }
            if n_ != 0 {
                if m_ != 0 {
                    pos[nu] = cur.value * mw[(n_ + md * (m_ - 1)) as usize];
                }
                if m_ != n_ {
                    neg[nu] = -cur.value * mw[(n_ + md * m_) as usize];
                }
            }
            ns -= 1;
            next_node = stack[ns];
            from_child = node;
            unwind!(pfeat, from_flag);
            continue;
        }

        let feat = cur.feat as usize;
        next_xnode = pick(xdir[nu], cl, cr, cd);
        next_rnode = pick(rdir[nu], cl, cr, cd);

        if next_xnode >= 0 {
            if next_xnode != next_rnode {
                flag[next_xnode as usize] = FROM_X_NOT_R;
                flag[next_rnode as usize] = FROM_R_NOT_X;
            } else {
                flag[next_xnode as usize] = FROM_NEITHER;
            }
        }

        if from_child == -1 {
            stack[ns] = node;
            ns += 1;
            next_node = -1;
            if hist[feat] > 0 {
                next_node = next_xnode;
                hist[feat] += 1;
            } else if hist[feat] < 0 {
                next_node = next_rnode;
                hist[feat] -= 1;
            }
            if next_node < 0 && next_xnode == next_rnode {
                next_node = next_xnode;
            }
            if next_node >= 0 {
                continue;
            }
            next_node = cl;
            if next_rnode == next_node {
                n_ += 1;
                hist[feat] -= 1;
            } else if next_xnode == next_node {
                m_ += 1;
                n_ += 1;
                hist[feat] += 1;
            }
            from_child = -1;
            continue;
        }

        // arriving from a child
        next_node = -1;
        if next_rnode == next_xnode || hist[feat] != 0 {
            next_node = pnode;
        }
        if next_node >= 0 {
            if node == 0 {
                break;
            }
            pos[nu] = pos[from_child as usize];
            neg[nu] = neg[from_child as usize];
            from_child = node;
            ns -= 1;
            unwind!(pfeat, from_flag);
            continue;
        } else if from_child == cl {
            stack[ns] = node;
            ns += 1;
            next_node = cr;
            if next_xnode == next_node {
                m_ += 1;
                n_ += 1;
                hist[feat] += 1;
            } else if next_rnode == next_node {
                n_ += 1;
                hist[feat] -= 1;
            }
            from_child = -1;
            continue;
        } else if from_child == cr {
            let (mut pos_x, mut neg_x, mut pos_r, mut neg_r) = (0f32, 0f32, 0f32, 0f32);
            let (clu, cru) = (cl as usize, cr as usize);
            if next_xnode == cr && next_rnode == cl {
                pos_x = pos[cru];
                neg_x = neg[cru];
                pos_r = pos[clu];
                neg_r = neg[clu];
            } else if next_xnode == cl && next_rnode == cr {
                pos_x = pos[clu];
                neg_x = neg[clu];
                pos_r = pos[cru];
                neg_r = neg[cru];
            }
            out[feat] += (pos_x + neg_r) as f64;
            pos[nu] = pos_x + pos_r;
            neg[nu] = neg_x + neg_r;
            if node == 0 {
                break;
            }
            ns -= 1;
            next_node = stack[ns];
            from_child = node;
            unwind!(pfeat, from_flag);
            continue;
        }
    }
}

/// dense_independent for one row x and one output index (tree-outer, reference-inner loop order;
/// per-reference accumulators keep the per-reference rescaling exact). `rdirs` holds precomputed
/// directions of every reference: [tree][ref][node].
#[allow(clippy::too_many_arguments)]
pub fn independent_row(
    e: &Ensemble,
    nodes: &[Node],
    mw: &[f32],
    oind: usize,
    x: &[f64],
    xm: &[bool],
    y_i: f64,
    rdirs: &[u8],
    num_r: usize,
    m: usize,
    margin_x: f64,
    margins_r: &[f64],
    transform: Option<Transform>,
    out_row: &mut [f64],
    tmps: &mut [f64],
    xdir: &mut [u8],
    s: &mut Scratch,
) {
    let nout = e.nout;
    let mn = e.max_nodes;
    tmps.fill(0.0);
    for k in 0..e.tree_limit {
        let tree = &nodes[k * mn..(k + 1) * mn];
        directions(tree, x, xm, xdir);
        let rbase = k * num_r * mn;
        for j in 0..num_r {
            let rdir = &rdirs[rbase + j * mn..rbase + (j + 1) * mn];
            tree_shap_indep(e.max_depth, m, xdir, rdir, &mut tmps[j * (m + 1)..(j + 1) * (m + 1)], mw, tree, s);
        }
    }
    let mut rescale = 1.0f64;
    for j in 0..num_r {
        let tmp = &tmps[j * (m + 1)..(j + 1) * (m + 1)];
        if let Some(tf) = transform {
            let margin_r = margins_r[j];
            if margin_x == margin_r {
                rescale = 1.0;
            } else {
                rescale = (tf(margin_x, y_i) - tf(margin_r, y_i)) / (margin_x - margin_r);
            }
        }
        for k in 0..m {
            out_row[k * nout + oind] += tmp[k] * rescale;
        }
        if let Some(tf) = transform {
            out_row[m * nout + oind] += tf(e.base_offset[oind] + tmp[m], 0.0);
        } else {
            out_row[m * nout + oind] += e.base_offset[oind] + tmp[m];
        }
    }
    for j in 0..m + 1 {
        out_row[j * nout + oind] /= num_r as f64;
    }
}
