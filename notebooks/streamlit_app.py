import sys
import warnings
from pathlib import Path

# Ensure project root is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from scipy.io import loadmat

# Set matplotlib backend to non-interactive
matplotlib.use("Agg")

try:
    import mne
    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

# Constants and file paths
HO_RLSL_RESULTS_FILE = PROJECT_ROOT / "outputs/ho_rlsl_results.npz"
PRECOMPUTED_FCCA_ALL_FILE = PROJECT_ROOT / "outputs/fcca_results/precomputed_fcca_all.npz"
MASTER_PREP_REPORT = PROJECT_ROOT / "outputs/processed_1024/master_preprocessing_report.csv"
BALANCED_EPOCHS_DIR = PROJECT_ROOT / "outputs/processed_1024/02_balanced_theta_epochs"

TENSOR_FILES = {
    "incorrect": PROJECT_ROOT / "outputs/tensor_4d_1024/tensor_incorrect_4d.npy",
    "correct": PROJECT_ROOT / "outputs/tensor_4d_1024/tensor_correct_4d.npy"
}

ALGO_CP_FILES = {
    "HO-RLSL": PROJECT_ROOT / "outputs/fcca_results/HO_RLSL/HORLSL_ChangePoints.npy",
    "HOSVD": PROJECT_ROOT / "outputs/fcca_results/HOSVD/HOSVD_ChangePoints.npy",
    "PELT": PROJECT_ROOT / "outputs/fcca_results/PELT/PELT_ChangePoints.npy",
    "DMD": PROJECT_ROOT / "outputs/fcca_results/DMD/DMD_ChangePoints.npy",
    "CP-Tracking": PROJECT_ROOT / "outputs/fcca_results/CP_Tracking/CP_ChangePoints.npy"
}

# Color palette for communities
COMMUNITY_PALETTE = (
    "#1f77b4",  # Muted Blue
    "#d62728",  # Muted Red
    "#2ca02c",  # Muted Green
    "#9467bd",  # Purple
    "#ff7f0e",  # Orange
    "#17becf",  # Cyan
    "#e377c2",  # Pink
    "#8c564b",  # Brown
    "#bcbd22",  # Olive
    "#7f7f7f",  # Grey
)

# Page configuration
st.set_page_config(
    page_title="Brain Connectivity Tensor Tracking Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling injection
st.markdown("""
<style>
    /* Premium slate-dark styling override */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stSidebar {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card h4 {
        margin: 0;
        color: #8b949e;
        font-size: 0.9rem;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #58a6ff;
        font-size: 1.6rem;
        font-weight: bold;
    }
    .formula-card {
        background-color: #1f242c;
        border-left: 5px solid #58a6ff;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .step-container {
        border-left: 2px dashed #30363d;
        padding-left: 20px;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FCCA MATHEMATICAL ENGINE (Structure-preserving port of cluster_ern_fcca_paper.m)
# -----------------------------------------------------------------------------

EPS = np.finfo(float).eps

def weighted_modularity(A: np.ndarray, labels: np.ndarray) -> float:
    A = sanitize_adjacency(A)
    labels = np.asarray(labels).reshape(-1)
    m2 = np.sum(A)
    if m2 <= EPS:
        return 0.0

    degree = np.sum(A, axis=1)
    same_cluster = labels[:, None] == labels[None, :]
    expected = np.outer(degree, degree) / m2
    q = np.sum((A - expected) * same_cluster) / m2
    return float(q)

def stable_unique(values):
    arr = np.asarray(values).reshape(-1)
    seen = set()
    out = []
    for value in arr:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return np.asarray(out)

def relabel(labels):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    unique_labels = stable_unique(labels)
    out = np.zeros_like(labels)
    for idx, label in enumerate(unique_labels, start=1):
        out[labels == label] = idx
    return out

def clusters_to_labels(clusters, n):
    labels = np.zeros(n, dtype=int)
    for idx, cluster in enumerate(clusters, start=1):
        labels[cluster] = idx
    return relabel(labels)

def fiedler_bipartition(W):
    n = W.shape[0]
    if n == 1:
        return np.array([1], dtype=int)
    if n == 2:
        return np.array([1, 2], dtype=int)

    W = sanitize_adjacency(W)
    degree = np.sum(W, axis=1)
    valid = degree > EPS
    if np.count_nonzero(valid) < 2:
        # Fallback half-split
        labels = np.ones(n, dtype=int)
        labels[n // 2 :] = 2
        return labels

    inv_sqrt_degree = np.zeros(n, dtype=float)
    inv_sqrt_degree[valid] = 1.0 / np.sqrt(degree[valid])
    L = np.eye(n) - (inv_sqrt_degree[:, None] * W) * inv_sqrt_degree[None, :]
    L = (L + L.T) / 2.0

    evals, V = np.linalg.eigh(L)
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    V = np.real(V[:, order])
    
    fiedler_candidates = np.where(evals > 1e-10)[0]
    if fiedler_candidates.size == 0:
        labels = np.ones(n, dtype=int)
        labels[n // 2 :] = 2
        return labels

    u = V[:, fiedler_candidates[0]]
    sort_order = np.argsort(u, kind="mergesort")
    u_sorted = u[sort_order]
    gaps = np.diff(u_sorted)
    if gaps.size == 0:
        labels = np.ones(n, dtype=int)
        labels[n // 2 :] = 2
        return labels

    gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[gap_idx])
    if max_gap <= EPS:
        labels = np.ones(n, dtype=int)
        labels[n // 2 :] = 2
        return labels

    labels = np.zeros(n, dtype=int)
    split = gap_idx + 1
    labels[sort_order[:split]] = 1
    labels[sort_order[split:]] = 2
    return labels

def fcca_split_cluster(graphs, nodes):
    m = graphs.shape[2]
    p = nodes.size
    cooccur = np.zeros((p, p), dtype=float)

    for graph_idx in range(m):
        W = graphs[:, :, graph_idx][np.ix_(nodes, nodes)]
        local_labels = fiedler_bipartition(W)
        cooccur += (local_labels[:, None] == local_labels[None, :]).astype(float)

    cooccur /= m
    consensus_labels = fiedler_bipartition(cooccur)
    part_a = nodes[consensus_labels == 1]
    part_b = nodes[consensus_labels == 2]
    return part_a, part_b, cooccur

def histcounts_discrete(values, n_bins):
    values = np.rint(np.asarray(values)).astype(int).reshape(-1)
    values = values[(values >= 1) & (values <= n_bins)]
    if values.size == 0:
        return np.zeros(n_bins, dtype=float)
    return np.bincount(values, minlength=n_bins + 1)[1 : n_bins + 1].astype(float)

def normalize_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = np.sum(counts)
    if total <= 0:
        return np.ones_like(counts, dtype=float) / counts.size
    return counts / total

def entropy_base2(p):
    p = np.asarray(p, dtype=float).reshape(-1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def kl_base2(p, q):
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

def jensen_shannon(p, q):
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    js = 0.5 * kl_base2(p, m) + 0.5 * kl_base2(q, m)
    return float(max(0.0, min(1.0, js)))

def rank_distributions_for_cluster(rank_bins, labels, c, n):
    in_cluster = labels == c
    out_cluster = ~in_cluster

    node_i, node_j = np.where(~np.eye(n, dtype=bool))
    intra_counts = np.zeros(n, dtype=float)
    inter_counts = np.zeros(n, dtype=float)

    for graph_idx in range(rank_bins.shape[2]):
        R = rank_bins[:, :, graph_idx].astype(float)
        ranks = R[node_i, node_j]

        intra_mask = in_cluster[node_i] & in_cluster[node_j]
        inter_mask = in_cluster[node_i] & out_cluster[node_j]

        intra_counts += histcounts_discrete(ranks[intra_mask], n)
        inter_counts += histcounts_discrete(ranks[inter_mask], n)

    return normalize_counts(intra_counts), normalize_counts(inter_counts)

def fcca_quality(rank_bins, labels):
    labels = relabel(labels)
    n = labels.size
    k = int(np.max(labels))
    cluster_h = np.zeros(k, dtype=float)
    cluster_c = np.zeros(k, dtype=float)

    for c in range(1, k + 1):
        pintra, pinter = rank_distributions_for_cluster(rank_bins, labels, c, n)
        entropy_intra = entropy_base2(pintra)
        alpha = np.sum(labels == c) / n
        cluster_h[c - 1] = alpha * max(0.0, 1.0 - entropy_intra / max(np.log2(n), EPS))
        cluster_c[c - 1] = jensen_shannon(pintra, pinter)

    Hbar = float(np.mean(cluster_h))
    Cbar = float(np.mean(cluster_c))
    if Hbar <= 0 or Cbar <= 0:
        U = 0.0
    else:
        U = 2.0 / (1.0 / Hbar + 1.0 / Cbar)
    return U, Hbar, Cbar, cluster_h, cluster_c

def compute_rank_bins(graphs):
    n, _, m = graphs.shape
    rank_bins = np.zeros((n, n, m), dtype=np.uint16)

    for graph_idx in range(m):
        W = sanitize_adjacency(graphs[:, :, graph_idx])
        R = np.zeros((n, n), dtype=np.uint16)

        for i in range(n):
            weights = W[i, :].copy()
            weights[i] = -np.inf
            order = np.argsort(-weights, kind="mergesort")

            ranks = np.zeros(n, dtype=np.uint16)
            ranks[order] = np.arange(1, n + 1, dtype=np.uint16)
            ranks[i] = 0
            R[i, :] = ranks

        rank_bins[:, :, graph_idx] = R

    return rank_bins

def choose_cluster_to_split(graphs, rank_bins, clusters, min_size=3):
    if len(clusters) == 1:
        return 0 if clusters[0].size >= 2 * min_size else None

    n_clusters = len(clusters)
    counts = np.zeros(n_clusters, dtype=float)
    labels = clusters_to_labels(clusters, graphs.shape[0])
    _, _, _, cluster_h, cluster_c = fcca_quality(rank_bins, labels)

    for gamma in np.arange(0.0, 1.0 + 1e-12, 0.1):
        zeta = np.full(n_clusters, np.inf)
        for idx, cluster in enumerate(clusters):
            if cluster.size >= 2 * min_size:
                zeta[idx] = cluster_c[idx] * gamma + cluster_h[idx] * (1.0 - gamma)
        bad_idx = int(np.argmin(zeta))
        counts[bad_idx] += 1

    for idx, cluster in enumerate(clusters):
        if cluster.size < 2 * min_size:
            counts[idx] = -np.inf

    if np.all(~np.isfinite(counts)):
        return None
    return int(np.argmax(counts))

def run_fcca_model_selection_exact(graphs, k_range):
    n = graphs.shape[0]
    k_max = int(max(k_range))
    rank_bins = compute_rank_bins(graphs)

    clusters = [np.arange(n, dtype=int)]
    hierarchy = [None] * (k_max + 1)
    quality_by_k = np.full(k_max + 1, np.nan)
    homogeneity_by_k = np.full(k_max + 1, np.nan)
    completeness_by_k = np.full(k_max + 1, np.nan)
    modularity_by_k = np.full(k_max + 1, np.nan)
    cooccur_by_k = [None] * (k_max + 1)

    for k in range(2, k_max + 1):
        split_idx = choose_cluster_to_split(graphs, rank_bins, clusters)
        if split_idx is None:
            break

        nodes = clusters[split_idx]
        part_a, part_b, cooccur = fcca_split_cluster(graphs, nodes)
        if part_a.size == 0 or part_b.size == 0 or part_a.size < 2 or part_b.size < 2:
            break

        clusters.pop(split_idx)
        clusters.extend([part_a, part_b])
        # Sort clusters by first element
        clusters = [clusters[idx] for idx in np.argsort([np.min(c) for c in clusters])]

        labels = clusters_to_labels(clusters, n)
        U, Hbar, Cbar, _, _ = fcca_quality(rank_bins, labels)
        Q = weighted_modularity(np.mean(graphs, axis=2), labels)

        hierarchy[k] = [c.copy() for c in clusters]
        quality_by_k[k] = U
        homogeneity_by_k[k] = Hbar
        completeness_by_k[k] = Cbar
        modularity_by_k[k] = Q

        # cooccurrence từ full labels (đồng bộ với FCCA script)
        cooccur_by_k[k] = (labels[:, None] == labels[None, :]).astype(float)

    valid_k = np.array([k for k in k_range if np.isfinite(quality_by_k[k])], dtype=int)
    if valid_k.size == 0:
        raise RuntimeError("FCCA failed to produce any valid k")

    # Model selection based on selection metric U
    best_pos = int(np.nanargmax(quality_by_k[valid_k]))
    best_k = int(valid_k[best_pos])
    best_clusters = hierarchy[best_k]
    labels = clusters_to_labels(best_clusters, n)

    quality_columns = ["k", "SelectionScore", "U", "H", "C", "Q"]
    quality_rows = []
    for k in valid_k:
        quality_rows.append([
            float(k),
            float(quality_by_k[k]),
            float(quality_by_k[k]),
            float(homogeneity_by_k[k]),
            float(completeness_by_k[k]),
            float(modularity_by_k[k])
        ])
    quality_array = np.array(quality_rows, dtype=float)

    return {
        "bestK": best_k,
        "labels": labels,
        "cooccurrence": cooccur_by_k[best_k] if cooccur_by_k[best_k] is not None else np.eye(n),
        "qualityTableArray": quality_array,
        "qualityTableColumns": quality_columns,
    }


@st.cache_data(show_spinner="Running live Fiedler Consensus community Splits (FCCA)...")
def run_exact_paper_fcca_live(X_clean, frames_tuple):
    frames = np.asarray(frames_tuple, dtype=int)
    n_subjects, n_nodes, _, _ = X_clean.shape
    
    # Extract interval graphs (nodes x nodes x subjects)
    graphs = np.zeros((n_nodes, n_nodes, n_subjects), dtype=float)
    for s in range(n_subjects):
        W = np.mean(np.take(X_clean[s], frames, axis=-1), axis=-1)
        graphs[:, :, s] = sanitize_adjacency(W)
        
    k_range = [2, 3, 4, 5, 6]
    fcca = run_fcca_model_selection_exact(graphs, k_range)
    mean_adj = graphs.mean(axis=2)
    
    return {
        "labels": fcca["labels"],
        "best_k": fcca["bestK"],
        "cooccurrence": fcca["cooccurrence"],
        "quality_table": fcca["qualityTableArray"],
        "mean_adjacency": mean_adj
    }


# -----------------------------------------------------------------------------
# Streamlit Setup and UI helpers
# -----------------------------------------------------------------------------

# Cached data loaders
@st.cache_data(show_spinner="Loading Master Preprocessing Report...")
def load_master_report():
    if MASTER_PREP_REPORT.exists():
        return pd.read_csv(MASTER_PREP_REPORT)
    return None

@st.cache_data(show_spinner="Loading Electrode Layout...")
def load_electrode_layout():
    bids_dir = PROJECT_ROOT / "data" / "ERN_Raw_Data_BIDS-Compatible"
    tsv_files = sorted(list(bids_dir.glob("sub-*/eeg/*_electrodes.tsv")))
    if tsv_files:
        path = tsv_files[0]
        try:
            names = []
            x_vals = []
            y_vals = []
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                name_idx = header.index("name")
                x_idx = header.index("x")
                y_idx = header.index("y")
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) > max(name_idx, x_idx, y_idx):
                        x_str = parts[x_idx].strip()
                        y_str = parts[y_idx].strip()
                        if x_str.lower() not in ["n/a", "nan", ""] and y_str.lower() not in ["n/a", "nan", ""]:
                            names.append(parts[name_idx])
                            x_vals.append(float(x_str))
                            y_vals.append(float(y_str))
            
            names = np.asarray(names, dtype=object)
            n_nodes = 30
            names = names[:n_nodes]
            x_vals = np.asarray(x_vals[:n_nodes])
            y_vals = np.asarray(y_vals[:n_nodes])
            
            xy = np.column_stack([-y_vals, x_vals])
            radius = np.max(np.sqrt(np.sum(xy**2, axis=1)))
            if radius > 1e-12:
                xy = 0.96 * xy / radius
            return xy, [str(name) for name in names]
        except Exception as e:
            pass
            
    # Fallback to circle layout
    n_nodes = 30
    theta = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, n_nodes + 1)[:-1]
    xy = np.column_stack([0.88 * np.cos(theta), 0.88 * np.sin(theta)])
    names = [f"N{i}" for i in range(1, n_nodes + 1)]
    return xy, names

@st.cache_data(show_spinner="Loading HO-RLSL results...")
def load_ho_rlsl_results():
    if not HO_RLSL_RESULTS_FILE.exists():
        return None
    try:
        with np.load(HO_RLSL_RESULTS_FILE, allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}
    except Exception as e:
        st.error(f"Error loading HO-RLSL results NPZ: {e}")
        return None

@st.cache_data(show_spinner="Loading pre-computed FCCA results...")
def load_precomputed_fcca_all():
    if not PRECOMPUTED_FCCA_ALL_FILE.exists():
        return None
    try:
        with np.load(PRECOMPUTED_FCCA_ALL_FILE, allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}
    except Exception as e:
        st.error(f"Error loading precomputed FCCA NPZ: {e}")
        return None

@st.cache_data(show_spinner="Loading Paper MAT Results...")
def load_paper_method_results(cp_method):
    folder_name = cp_method.replace("-", "_")
    path = PROJECT_ROOT / "outputs" / "fcca_results" / "fcca_ern_by_method" / folder_name / "fcca_paper_interval_results.mat"
    if not path.exists():
        return None
    try:
        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return data["results"]
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_algo_change_points(algo, condition):
    cp_file = ALGO_CP_FILES.get(algo)
    if cp_file is None or not cp_file.exists():
        return None
    try:
        cp_dict = np.load(str(cp_file), allow_pickle=True).item()
        cond_key = "1024Hz_ERN" if condition == "incorrect" else "1024Hz_CRN"
        return cp_dict.get(cond_key, None)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_subject_list():
    if BALANCED_EPOCHS_DIR.exists():
        return sorted([f.name.split("_")[0] for f in BALANCED_EPOCHS_DIR.glob("sub-*_theta_balanced-epo.fif")])
    return []

@st.cache_data(show_spinner="Extracting Subject Waveform...")
def load_subject_erp_waveform(subject_name, channel_name):
    if not HAS_MNE:
        return None, None, "MNE is not installed. Waveform cannot be read."
    
    fif_file = BALANCED_EPOCHS_DIR / f"{subject_name}_theta_balanced-epo.fif"
    if not fif_file.exists():
        return None, None, f"Refined epoch file for {subject_name} not found."
    
    try:
        epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
        if channel_name not in epochs.ch_names:
            return None, None, f"Channel {channel_name} not found in this subject."
        
        ch_idx = epochs.ch_names.index(channel_name)
        times_ms = epochs.times * 1000.0
        
        # Extract mean waveforms
        correct_wave = epochs["Correct"].get_data(verbose=False)[:, ch_idx, :].mean(axis=0)
        incorrect_wave = epochs["Incorrect"].get_data(verbose=False)[:, ch_idx, :].mean(axis=0)
        
        # Apply baseline correction
        baseline_mask = (times_ms >= -200.0) & (times_ms <= 0.0)
        if baseline_mask.any():
            correct_wave -= np.mean(correct_wave[baseline_mask])
            incorrect_wave -= np.mean(incorrect_wave[baseline_mask])
        
        correct_wave *= 1e6
        incorrect_wave *= 1e6
        
        return times_ms, {"correct": correct_wave, "incorrect": incorrect_wave}, None
    except Exception as e:
        return None, None, f"Error reading epoch file: {e}"


# Basic Adjacency Processing Helpers
def sanitize_adjacency(A):
    A = np.asarray(A, dtype=float).copy()
    A[~np.isfinite(A)] = 0
    A = (A + A.T) / 2
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    return A

def threshold_to_edges(A, max_plot_edges):
    A = sanitize_adjacency(A)
    n = A.shape[0]
    row, col = np.triu_indices(n, 1)
    weight = A[row, col]
    positive = weight > 0
    row = row[positive]
    col = col[positive]
    weight = weight[positive]
    
    order = np.argsort(-weight, kind="mergesort")
    row = row[order]
    col = col[order]
    weight = weight[order]
    
    n_edges = min(max_plot_edges, weight.size)
    thresholded = np.zeros_like(A)
    if n_edges > 0:
        for idx in range(n_edges):
            thresholded[row[idx], col[idx]] = weight[idx]
            thresholded[col[idx], row[idx]] = weight[idx]
    return thresholded, row, col, weight, n_edges

def nodes_without_positive_intra_cluster_edge(A, labels):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    n = labels.size
    isolated_nodes = []
    all_nodes = np.arange(n)

    for node in range(n):
        same_cluster = all_nodes[(labels == labels[node]) & (all_nodes != node)]
        if same_cluster.size == 0:
            isolated_nodes.append(node)
            continue
        if np.nansum(A[node, same_cluster]) <= 1e-12:
            isolated_nodes.append(node)

    return np.asarray(isolated_nodes, dtype=int)

def node_metric_table(A, labels, A_thr, channel_names):
    return pd.DataFrame(
        {
            "Channel": channel_names,
            "Community": labels.astype(int),
            "Connectivity Strength": A.sum(axis=1),
            "Thresholded Degree": (A_thr > 0).sum(axis=1),
            "Thresholded Strength": A_thr.sum(axis=1),
        }
    )

# Visualization Helpers
def draw_head_outline(ax, xy):
    center = xy.mean(axis=0)
    radius = max(np.max(np.linalg.norm(xy - center, axis=1)) * 1.08, 1.0)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color="#8b949e",
        linewidth=1.0,
        zorder=0,
    )
    # Nose
    ax.plot(
        [center[0] - 0.10 * radius, center[0], center[0] + 0.10 * radius],
        [center[1] + radius, center[1] + 1.12 * radius, center[1] + radius],
        color="#8b949e",
        linewidth=1.0,
        zorder=0,
    )

def community_color_map(labels):
    labels = np.asarray(labels, dtype=int)
    communities = sorted(int(label) for label in np.unique(labels))
    return {
        community: COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
        for idx, community in enumerate(communities)
    }

def community_edge_style(i, j, labels):
    labels = np.asarray(labels, dtype=int)
    color_map = community_color_map(labels)
    if labels[i] == labels[j]:
        return color_map[int(labels[i])], 0.65, 1.8
    return "#30363d", 0.15, 0.8

def add_network_legend(ax, labels):
    colors = community_color_map(labels)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
            label=f"Community {community}",
        )
        for community, color in colors.items()
    ]
    ax.legend(
        handles=handles,
        title="Consensus Communities",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

def community_node_colors(labels, n_nodes):
    if labels is None:
        return ["#ff8c1a"] * n_nodes
    labels = np.asarray(labels, dtype=int)
    colors = community_color_map(labels)
    return [colors[int(label)] for label in labels]

def plot_paper_electrode_network(A, labels, max_plot_edges, show_node_labels, xy, names):
    # Retrieve sorted edge weights and filter by max count
    A_san = sanitize_adjacency(A)
    A_thr, row, col, weight, n_edges = threshold_to_edges(A_san, max_plot_edges)
    n_nodes = A_san.shape[0]
    
    node_colors = community_node_colors(labels, n_nodes)
    
    # Identify isolated nodes in the full adjacency matrix A and paint them gray
    isolated_nodes = nodes_without_positive_intra_cluster_edge(A_san, labels)
    for idx in isolated_nodes:
        node_colors[idx] = "#727272" # Slate gray for isolated nodes

    fig, ax = plt.subplots(figsize=(6.0, 5.6), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    draw_head_outline(ax, xy)

    # Plot thresholded edges with paper-compliant grayscale styling
    if n_edges > 0:
        selected_weights = weight[:n_edges]
        w_min = selected_weights.min()
        w_max = selected_weights.max()
        w_range = w_max - w_min
        if w_range <= 1e-12:
            scaled = np.ones_like(selected_weights)
        else:
            scaled = (selected_weights - w_min) / w_range
            
        for edge_idx in range(n_edges - 1, -1, -1):
            # Grayscale shade from 0.18 (dark) to 0.73 (light)
            shade = 0.18 + 0.55 * (1.0 - scaled[edge_idx])
            edge_color = (shade, shade, shade)
            linewidth = 0.35 + 1.7 * scaled[edge_idx]
            ax.plot(
                [xy[row[edge_idx], 0], xy[col[edge_idx], 0]],
                [xy[row[edge_idx], 1], xy[col[edge_idx], 1]],
                color=edge_color,
                linewidth=linewidth,
                zorder=1,
            )

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=node_colors,
        s=130,
        edgecolors="#30363d",
        linewidths=1.0,
        zorder=3,
    )

    if show_node_labels:
        for idx, (x_pos, y_pos) in enumerate(xy):
            ax.text(
                x_pos,
                y_pos,
                names[idx],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if labels[idx] in [1, 2, 5] else "black",
                zorder=4,
            )

    ax.set_title(f"Scalp Electrode Connectivity Map\n(top {n_edges} strongest edges)", color="white", fontsize=10)
    add_network_legend(ax, labels)
    ax.set_aspect("equal")
    ax.axis("off")
    pad_x = max((xy[:, 0].max() - xy[:, 0].min()) * 0.12, 0.12)
    pad_y = max((xy[:, 1].max() - xy[:, 1].min()) * 0.15, 0.15)
    ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
    ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
    fig.tight_layout()
    return fig, A_thr


def plot_connectivity_heatmap(A, labels, sort_by_community, channel_names):
    A = sanitize_adjacency(A)
    order = np.arange(A.shape[0])
    if labels is not None and sort_by_community:
        order = np.argsort(labels)
        A = A[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(6.0, 5.2), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    im = ax.imshow(A, cmap="viridis", interpolation="nearest")
    ax.set_title("Connectivity Matrix Heatmap", color="white", fontsize=10)
    ax.set_xlabel("Nodes (Sorted)", color="#8b949e")
    ax.set_ylabel("Nodes (Sorted)", color="#8b949e")
    ax.set_xticks(np.arange(A.shape[0]))
    ax.set_yticks(np.arange(A.shape[0]))
    ax.set_xticklabels([channel_names[idx] for idx in order], rotation=90, fontsize=7, color="#8b949e")
    ax.set_yticklabels([channel_names[idx] for idx in order], fontsize=7, color="#8b949e")
    
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    fig.tight_layout()
    return fig


def main():
    st.title("🧠 Brain Connectivity Subspace Tracking & Community Dynamics")
    st.caption("Step-by-step Interactive Walkthrough of the True Paper Pipeline")

    # Data loading
    report = load_master_report()
    ho_rlsl = load_ho_rlsl_results()
    fcca_all = load_precomputed_fcca_all()

    # Integrity Check
    if ho_rlsl is None:
        st.error("Missing critical HO-RLSL results file: ho_rlsl_results.npz. "
                 "Please make sure it exists in outputs/")
        st.stop()

    # Extract time axis and coordinates dynamically
    times_ms = np.linspace(-1000.0, 1000.0, ho_rlsl["low_rank"].shape[-1])
    xy, channel_names = load_electrode_layout()
    n_nodes = len(channel_names)

    # Sidebar Step Selector
    with st.sidebar:
        st.header("Pipeline Navigation")
        step = st.radio(
            "Select Execution Step:",
            [
                "Pipeline Overview",
                "Steps 1 & 2: Preprocessing & Trial Balancing",
                "Step 3: Phase Locking Value Tensors",
                "Step 4: HO-RLSL Low-Rank Extraction",
                "Step 5: Change Points & FCCA splits"
            ]
        )

        st.markdown("---")
        st.header("Global Controls")
        condition = st.selectbox(
            "Experimental Condition:",
            ["Incorrect (ERN)", "Correct (CRN)"],
            index=0,
            help="ERN condition corresponds to error trials (Incorrect), CRN corresponds to correct trials."
        )
        cond_key = "incorrect" if "Incorrect" in condition else "correct"

        # Change Point selection logic
        if step in ["Step 4: HO-RLSL Low-Rank Extraction", "Step 5: Change Points & FCCA splits"]:
            st.header("Tracking Algorithm")
            cp_method = st.selectbox(
                "Change Point Method:",
                ["HO-RLSL", "HOSVD", "PELT", "DMD", "CP-Tracking"],
                index=0,
                help="Select the tracking algorithm to define connection change points."
            )
        else:
            cp_method = "HO-RLSL"

        # Adaptive Controls based on step selection
        if step in ["Step 3: Phase Locking Value Tensors", "Step 4: HO-RLSL Low-Rank Extraction", "Step 5: Change Points & FCCA splits"]:
            st.subheader("Visualization Parameters")
            max_plot_edges = st.slider("Maximum Plot Edges", 10, 100, 35, 5, help="Select maximum number of strongest connections to draw on the network.")
            show_labels = st.checkbox("Show Electrode Labels", value=True)
            sort_heatmap = st.checkbox("Sort Heatmap by Community", value=True)
        else:
            max_plot_edges = 35
            show_labels = True
            sort_heatmap = True

    # Render pages
    if step == "Pipeline Overview":
        render_overview(times_ms, channel_names, report)
        
    elif step == "Steps 1 & 2: Preprocessing & Trial Balancing":
        render_preprocessing_balancing(report, cond_key, channel_names)
        
    elif step == "Step 3: Phase Locking Value Tensors":
        render_plv_tensors(cond_key, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy)
        
    elif step == "Step 4: HO-RLSL Low-Rank Extraction":
        render_ho_rlsl(cond_key, ho_rlsl, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy)
        
    elif step == "Step 5: Change Points & FCCA splits":
        render_fcca_splits(cond_key, ho_rlsl, fcca_all, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy, cp_method)


def render_overview(times_ms, channel_names, report):
    st.subheader("End-to-End True Paper Pipeline Overview")
    st.markdown("""
    This application visualizes the step-by-step mathematical transformations of the true paper pipeline 
    as coordinated by `src/main.py`. The pipeline processes raw EEG signals into dynamic low-rank brain 
    connectivity networks and detects consensus community transitions.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pipeline Execution Flowchart")
        st.markdown("""
        1. **EEG Preprocessing**: CSD spatial filtering, 0.1-30Hz bandpass filtering, epoching.
        2. **Trial Balancing**: Balance epochs between Correct and Incorrect conditions to align signal-to-noise ratio.
        3. **PLV Connection Tensor**: Hilbert transform phase-locking computations for 4D connectivity tensors.
        4. **HO-RLSL Subspace Learning**: Separation of PLV tensor into Low-Rank and Sparse noise components.
        5. **Native Change Points**: Unsupervised detection of temporal network transitions.
        6. **Fiedler Consensus community Splits (FCCA)**: Community extraction and model selection ($k=2..6$).
        """)
    
    with col2:
        st.markdown("### Master Statistics")
        c1, c2 = st.columns(2)
        c1.metric("EEG Subjects", len(report) if report is not None else 40)
        c2.metric("EEG Sensors (Nodes)", len(channel_names))
        c3, c4 = st.columns(2)
        c3.metric("Sampling Rate", "1024 Hz")
        c4.metric("Time Windows (Epoch)", f"{len(times_ms)} points (-1.0 to +1.0s)")

    st.markdown("### Preprocessing Summary (All Subjects)")
    if report is not None:
        st.dataframe(report, use_container_width=True, hide_index=True)
    else:
        st.info("No preprocessing CSV report found at outputs/processed_1024/master_preprocessing_report.csv")


def render_preprocessing_balancing(report, cond_key, channel_names):
    st.subheader("Steps 1 & 2: Preprocessing EEG Data & Trial Balancing")
    
    # Text card explaining balancing
    st.markdown("""
    <div class="formula-card">
    <strong>Why Trial Balancing is Crucial:</strong><br>
    Phase Locking Value (PLV) connectivity measurements are highly sensitive to the number of epochs (trials) 
    used in computation. If one condition has 300 trials (e.g. Correct) and another has only 20 trials 
    (e.g. Incorrect), the connectivity values in the latter will be systematically inflated due to a higher noise floor. 
    To prevent this signal-to-noise ratio (SNR) bias, the trial count must be matched (balanced) between Correct 
    and Incorrect conditions for each subject before PLV computation.
    </div>
    """, unsafe_allow_html=True)

    if report is not None:
        col_chart, col_stats = st.columns([2, 1])
        with col_chart:
            st.markdown("#### Trial Distribution across Conditions")
            fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#161b22")
            ax.set_facecolor("#161b22")
            x = np.arange(len(report))
            width = 0.35
            ax.bar(x - width/2, report["correct"], width, label="Correct (CRN)", color="#1f77b4")
            ax.bar(x + width/2, report["incorrect"], width, label="Incorrect (ERN)", color="#d62728")
            ax.set_xlabel("Subjects", color="white")
            ax.set_ylabel("Trial Count", color="white")
            ax.set_title("Initial Unbalanced Trials per Subject", color="white")
            ax.set_xticks(x)
            ax.set_xticklabels(report["sub"], rotation=90, fontsize=6, color="#8b949e")
            ax.tick_params(colors="white")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
        with col_stats:
            st.markdown("#### Balancing Action")
            mean_correct = report["correct"].mean()
            mean_incorrect = report["incorrect"].mean()
            st.metric("Avg. Correct Trials", f"{mean_correct:.1f}")
            st.metric("Avg. Incorrect Trials", f"{mean_incorrect:.1f}")
            st.info("The trial matching pipeline selects a random subset of Correct trials equal to the "
                    "number of available Incorrect trials for each subject, achieving a 1:1 balance ratio.")

    # Dynamic EEG waveform viewer
    st.markdown("---")
    st.subheader("Interactive ERP Waveform Viewer")
    st.markdown("Load raw balanced EEG traces dynamically for individual subjects:")
    
    subjects = get_subject_list()
    if not subjects:
        st.warning("No balanced epoch FIF files found. Dynamic waveform viewer is disabled.")
    else:
        col1, col2, col3 = st.columns(3)
        selected_sub = col1.selectbox("Select Subject:", subjects)
        selected_ch = col2.selectbox("Select Channel (FCz/Cz recommended for ERN):", channel_names, index=channel_names.index("FCz") if "FCz" in channel_names else 0)
        
        times_ms, erp_data, err_msg = load_subject_erp_waveform(selected_sub, selected_ch)
        if err_msg:
            st.error(err_msg)
        else:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#161b22")
            ax.set_facecolor("#161b22")
            ax.plot(times_ms, erp_data["incorrect"], label="Incorrect (ERN)", color="#d62728", linewidth=1.5)
            ax.plot(times_ms, erp_data["correct"], label="Correct (CRN)", color="#1f77b4", linewidth=1.5)
            ax.axvline(0, color="#8b949e", linestyle="--", alpha=0.5, label="Stimulus Onset")
            ax.axhline(0, color="#8b949e", alpha=0.5)
            ax.set_xlabel("Time (ms)", color="white")
            ax.set_ylabel("Amplitude (μV / CSD Scaled)", color="white")
            ax.set_title(f"Balanced ERP Waveform for {selected_sub} (Channel {selected_ch})", color="white")
            ax.tick_params(colors="white")
            ax.set_xlim(-400, 600)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)


def render_plv_tensors(cond_key, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy):
    st.subheader("Step 3: Hilbert-based Phase Locking Value (PLV) Connectivity")
    
    st.markdown("""
    <div class="formula-card">
    <strong>Phase Locking Value (PLV) Formula:</strong><br>
    The phase locking value measures the phase consistency between two signals across trials over time:
    $$\\text{PLV}_{c_1, c_2}(t) = \\left| \\frac{1}{M} \\sum_{m=1}^M e^{j (\\phi_{c_1, m}(t) - \\phi_{c_2, m}(t))} \\right|$$
    Where $\\phi_{c, m}(t)$ is the instantaneous phase of channel $c$ on trial $m$ at time $t$, estimated via the Hilbert Transform of the theta-band (4-8 Hz) filtered EEG signal. PLV values range from 0 (random phase difference) to 1 (perfect phase synchronization).
    </div>
    """, unsafe_allow_html=True)

    tensor_path = TENSOR_FILES[cond_key]
    if not tensor_path.exists():
        st.error(f"PLV connectivity tensor not found at {tensor_path}. Please compute it first using src/main.py")
        st.stop()

    # Memory map load
    tensor = np.load(tensor_path, mmap_mode="r")
    n_nodes = tensor.shape[1]
    st.success(f"Successfully loaded 4D PLV connectivity tensor: **{tensor_path.name}**")
    
    col_dims, col_details = st.columns([1, 2])
    with col_dims:
        st.markdown("#### Tensor Dimensions")
        st.markdown(f"- **Subjects**: {tensor.shape[0]}")
        st.markdown(f"- **Nodes (Channels)**: {tensor.shape[1]} × {tensor.shape[2]}")
        st.markdown(f"- **Time Points**: {tensor.shape[3]}")
    with col_details:
        st.markdown("#### Subject-Averaged Global Connectivity Profile")
        global_mean = tensor.mean(axis=(0, 1, 2))
        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.plot(times_ms, global_mean, color="#58a6ff")
        ax.axvline(50, color="#d62728", linestyle="--", alpha=0.5, label="Expected ERN Peak")
        ax.set_xlabel("Time (ms)", color="white")
        ax.set_ylabel("Mean PLV", color="white")
        ax.tick_params(colors="white")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("PLV Spatial Connection Heatmap & Scalp Graph")
    time_idx = st.slider("Select Time Point for Spatial Matrix:", 0, len(times_ms)-1, int(len(times_ms)*0.53))
    st.metric("Approximate Time Window Center", f"{times_ms[time_idx]:.2f} ms")

    # Extract subject-averaged matrix slice
    A_mean = tensor[:, :, :, time_idx].mean(axis=0)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_heat = plot_connectivity_heatmap(A_mean, None, sort_heatmap, channel_names)
        st.pyplot(fig_heat)
    with col_b:
        fig_net, _ = plot_paper_electrode_network(A_mean, np.zeros(n_nodes), max_plot_edges, show_labels, xy, channel_names)
        st.pyplot(fig_net)


def render_ho_rlsl(cond_key, ho_rlsl, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy):
    st.subheader("Step 4: HO-RLSL Low-Rank Extraction & Subspace tracking")
    
    st.markdown("""
    <div class="formula-card">
    <strong>Higher-Order Recursive Low-Rank + Sparse Decomposition (HO-RLSL):</strong><br>
    HO-RLSL processes the noisy raw PLV tensor to isolate underlying structural networks. 
    It models the incoming connectivity measurement $M_t$ at time $t$ as:
    $$M_t = L_t + S_t$$
    Where $L_t$ is the low-rank component lying in a slowly varying Tucker subspace, and $S_t$ represents the sparse, transient connectivity noise. Subspaces are tracked recursively over time.
    </div>
    """, unsafe_allow_html=True)

    if cond_key != "incorrect":
        st.warning("HO-RLSL diagnostics and low-rank separation plots are only configured for the Incorrect (ERN) condition.")
        st.stop()

    low_rank = ho_rlsl["low_rank"]
    sparse = ho_rlsl["sparse"]

    # Global energy tracking
    st.markdown("### Dynamic Subspace Energy Tracking")
    lr_energy = np.linalg.norm(low_rank, axis=(1, 2)).mean(axis=0)
    sp_energy = np.linalg.norm(sparse, axis=(1, 2)).mean(axis=0)

    fig, ax = plt.subplots(figsize=(11, 2.5), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    ax.plot(times_ms, lr_energy, label="Low-Rank (Subspace Structure)", color="#58a6ff")
    ax.plot(times_ms, sp_energy, label="Sparse Outliers (Transient Noise)", color="#d62728")
    ax.set_xlabel("Time (ms)", color="white")
    ax.set_ylabel("Frobenius Norm Energy", color="white")
    ax.tick_params(colors="white")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Reconstruction Separation Viewer")
    time_idx = st.slider("Select Time Point for Reconstruction Matrix Slice:", 0, len(times_ms)-1, int(len(times_ms)*0.53))
    st.metric("Approximate Time Window Center", f"{times_ms[time_idx]:.2f} ms")

    # Load raw tensor slice to compare
    raw_tensor_path = TENSOR_FILES[cond_key]
    if raw_tensor_path.exists():
        raw_slice = np.load(raw_tensor_path, mmap_mode="r")[:, :, :, time_idx].mean(axis=0)
    else:
        raw_slice = low_rank[:, :, :, time_idx].mean(axis=0) + sparse[:, :, :, time_idx].mean(axis=0)

    lr_slice = low_rank[:, :, :, time_idx].mean(axis=0)
    sp_slice = sparse[:, :, :, time_idx].mean(axis=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### (1) Raw Connectivity ($M_t$)")
        fig1 = plot_connectivity_heatmap(raw_slice, None, False, channel_names)
        st.pyplot(fig1)
    with c2:
        st.markdown("#### (2) Low-Rank Subspace ($L_t$)")
        fig2 = plot_connectivity_heatmap(lr_slice, None, False, channel_names)
        st.pyplot(fig2)
    with c3:
        st.markdown("#### (3) Sparse Outliers ($S_t$)")
        fig3 = plot_connectivity_heatmap(sp_slice, None, False, channel_names)
        st.pyplot(fig3)


def render_fcca_splits(cond_key, ho_rlsl, fcca_all, times_ms, channel_names, max_plot_edges, show_labels, sort_heatmap, xy, cp_method):
    st.subheader("Step 5: Change Point Detection & Fiedler Consensus Community Splits (FCCA)")
    
    st.markdown("""
    <div class="formula-card">
    <strong>Fiedler Consensus community Splits (FCCA) and Model Selection:</strong><br>
    Change points slice the connectivity timecourse into Pre-ERN, ERN, and Post-ERN intervals. 
    FCCA aggregates subject connectivity community structures within each interval to construct a consensus 
    co-occurrence matrix. Spectral bipartitioning based on the Fiedler vector recursively splits consensus 
    networks, and consensus quality $U$ is tracked across different cluster counts $k$ to identify the optimal $k$.
    </div>
    """, unsafe_allow_html=True)

    # 1. Loading change points for selected algorithm
    st.markdown(f"### Subspace Tracking via {cp_method}")
    
    paper_results = None
    if cp_method == "HO-RLSL":
        if cond_key == "incorrect":
            cps_ms = [-860.4, -47.9, 139.6, 577.1, 702.1]
        else:
            cps_ms = [-860.0, -47.9, 139.6, 577.1]
        st.info(f"Using manual physiological intervals for HO-RLSL: {cps_ms}")
    else:
        if cond_key == "incorrect":
            paper_results = load_paper_method_results(cp_method)

        if paper_results is not None:
            cps_ms = list(paper_results.changePointTimesMs)
            st.success(f"Successfully loaded pre-computed paper MAT results for {cp_method}.")
        else:
            cps_ms = load_algo_change_points(cp_method, "incorrect" if cond_key == "incorrect" else "correct")
            if cps_ms is None:
                st.warning(f"No pre-computed change points found for {cp_method} under this condition. Falling back to paper change points.")
                cps_ms = [-15.6, 109.4]
            else:
                cps_ms = sorted(list(cps_ms))

    # Identify frames corresponding to change point milliseconds
    filtered_cps = []
    for ms in cps_ms:
        idx = int(np.argmin(np.abs(times_ms - ms)))
        if 0 < idx < len(times_ms):
            filtered_cps.append(idx)
    filtered_cps = sorted(set(filtered_cps))

    # 3. Interval & Results Loading
    st.markdown("---")
    st.subheader("Interval Selection & Community Visualizations")

    intervals_list = []
    loaded_precomputed = False

    if paper_results is not None and len(cps_ms) == 4:
        # Standard 3 paper intervals defined by the 4 change points
        intervals_list = [
            f"Pre-ERN ({cps_ms[0]:.0f} to {cps_ms[1]:.0f} ms)",
            f"ERN ({cps_ms[1]:.0f} to {cps_ms[2]:.0f} ms)",
            f"Post-ERN ({cps_ms[2]:.0f} to {cps_ms[3]:.0f} ms)"
        ]
        selected_interval_label = st.selectbox("Select Waveform Interval:", intervals_list, index=1)
        interval_idx = intervals_list.index(selected_interval_label)

        # Retrieve direct paper results from MAT file
        interval_keys = ["preERN", "ERN", "postERN"]
        key = interval_keys[interval_idx]
        interval_data = getattr(paper_results, key)
        
        consensus_matrix = interval_data.cooccurrence
        consensus_labels = interval_data.labels
        best_k = int(interval_data.bestK)
        mean_adj = interval_data.meanAdjacency
        quality_table = interval_data.qualityTable
        community_method = f"FCCA Paper MAT Results ({cp_method})"
        loaded_precomputed = True
    else:
        # Fallback interval generation
        if cp_method == "HO-RLSL" and cond_key == "incorrect":
            intervals_list = [
                f"Pre-ERN ({cps_ms[0]:.1f} to {cps_ms[1]:.1f} ms)",
                f"ERN ({cps_ms[1]:.1f} to {cps_ms[2]:.1f} ms)",
                f"Post-ERN ({cps_ms[2]:.1f} to {cps_ms[3]:.1f} ms)",
                f"Recovery ({cps_ms[3]:.1f} to {cps_ms[4]:.1f} ms)"
            ]
        elif len(cps_ms) == 4:
            intervals_list = [
                f"Pre-ERN ({cps_ms[0]:.1f} to {cps_ms[1]:.1f} ms)" if cond_key == "incorrect" else f"Pre-CRN ({cps_ms[0]:.1f} to {cps_ms[1]:.1f} ms)",
                f"ERN ({cps_ms[1]:.1f} to {cps_ms[2]:.1f} ms)" if cond_key == "incorrect" else f"CRN ({cps_ms[1]:.1f} to {cps_ms[2]:.1f} ms)",
                f"Post-ERN ({cps_ms[2]:.1f} to {cps_ms[3]:.1f} ms)" if cond_key == "incorrect" else f"Post-CRN ({cps_ms[2]:.1f} to {cps_ms[3]:.1f} ms)"
            ]
        elif len(cps_ms) == 2:
            intervals_list = [
                f"Pre-ERN ({times_ms[0]:.1f} to {cps_ms[0]:.1f} ms)",
                f"ERN ({cps_ms[0]:.1f} to {cps_ms[1]:.1f} ms)",
                f"Post-ERN ({cps_ms[1]:.1f} to {times_ms[-1]:.1f} ms)"
            ]
        else:
            # Fallback to simple boundaries
            boundaries = [0] + filtered_cps + [len(times_ms)]
            for idx in range(len(boundaries) - 1):
                start_ms = times_ms[boundaries[idx]]
                end_ms = times_ms[min(boundaries[idx+1], len(times_ms)-1)]
                intervals_list.append(f"Interval {idx+1} ({start_ms:.1f} to {end_ms:.1f} ms)")

        selected_interval_label = st.selectbox("Select Waveform Interval:", intervals_list, index=min(1, len(intervals_list)-1))
        interval_idx = intervals_list.index(selected_interval_label)

        # Precomputed file dùng Louvain/modularity (không phải FCCA paper),
        # nên chỉ dùng cho các algo không phải HO-RLSL.
        interval_names_map = {0: "pre_ern", 1: "ern", 2: "post_ern"}
        pre_name = interval_names_map.get(interval_idx)

        if cp_method != "HO-RLSL" and fcca_all is not None and pre_name is not None:
            matrix_key = f"{cond_key}_{cp_method}_{pre_name}_consensus_matrix"
            labels_key = f"{cond_key}_{cp_method}_{pre_name}_consensus_labels"
            mean_adj_key = f"{cond_key}_{cp_method}_{pre_name}_mean_adjacency"
            method_key = f"{cond_key}_{cp_method}_{pre_name}_community_method"

            if matrix_key in fcca_all and labels_key in fcca_all and mean_adj_key in fcca_all:
                consensus_matrix = fcca_all[matrix_key]
                consensus_labels = fcca_all[labels_key]
                mean_adj = fcca_all[mean_adj_key]
                best_k = len(np.unique(consensus_labels))
                quality_table = None

                method_val = fcca_all[method_key]
                if hasattr(method_val, "item"):
                    community_method = str(method_val.item())
                else:
                    community_method = str(method_val)

                loaded_precomputed = True

        if not loaded_precomputed:
            # Determine frames for live computation
            if cp_method == "HO-RLSL":
                # Load corresponding condition HO-RLSL low-rank file
                lr_file = PROJECT_ROOT / "outputs/fcca_results/HO_RLSL" / f"HORLSL_1024Hz_{'ERN' if cond_key == 'incorrect' else 'CRN'}_LowRank.npy"
                if lr_file.exists():
                    low_rank = np.load(lr_file)
                    if low_rank.shape[0] != 40: # subject first
                        low_rank = np.transpose(low_rank, (2, 0, 1, 3))
                else:
                    low_rank = ho_rlsl["low_rank"]
            else:
                low_rank = ho_rlsl["low_rank"]

            cp_frames = []
            for ms in cps_ms:
                idx = int(np.argmin(np.abs(times_ms - ms)))
                cp_frames.append(idx)
            
            if len(cps_ms) == 4:
                if interval_idx == 0:
                    frames_in_interval = np.arange(cp_frames[0], cp_frames[1])
                elif interval_idx == 1:
                    frames_in_interval = np.arange(cp_frames[1], cp_frames[2])
                else:
                    frames_in_interval = np.arange(cp_frames[2], cp_frames[3])
            elif len(cps_ms) == 5:
                if interval_idx == 0:
                    frames_in_interval = np.arange(cp_frames[0], cp_frames[1])
                elif interval_idx == 1:
                    frames_in_interval = np.arange(cp_frames[1], cp_frames[2])
                elif interval_idx == 2:
                    frames_in_interval = np.arange(cp_frames[2], cp_frames[3])
                else:
                    frames_in_interval = np.arange(cp_frames[3], cp_frames[4])
            elif len(cps_ms) == 2:
                if interval_idx == 0:
                    frames_in_interval = np.arange(0, cp_frames[0])
                elif interval_idx == 1:
                    frames_in_interval = np.arange(cp_frames[0], cp_frames[1])
                else:
                    frames_in_interval = np.arange(cp_frames[1], len(times_ms))
            else:
                boundaries = [0] + filtered_cps + [len(times_ms)]
                frames_in_interval = np.arange(boundaries[interval_idx], min(boundaries[interval_idx+1], len(times_ms)))
            
            # Run the live FCCA model selection!
            results = run_exact_paper_fcca_live(low_rank, tuple(frames_in_interval.tolist()))
            
            consensus_matrix = results["cooccurrence"]
            consensus_labels = results["labels"]
            best_k = results["best_k"]
            quality_table = results["quality_table"]
            mean_adj = results["mean_adjacency"]
            community_method = f"Live Fiedler Consensus splits (Exact Paper)"

    # Display best K
    st.info(f"Optimal Community Count selected for this interval: **k = {best_k}** ({community_method})")

    # Model Selection curves (Incorrect + HO-RLSL)
    if quality_table is not None:
        st.markdown("#### FCCA Model Selection Profile")
        fig_ms, ax_ms = plt.subplots(figsize=(9, 2.5), facecolor="#161b22")
        ax_ms.set_facecolor("#161b22")
        ks = quality_table[:, 0]
        u_score = quality_table[:, 2]
        h_score = quality_table[:, 3]
        c_score = quality_table[:, 4]
        q_score = quality_table[:, 5]
        
        ax_ms.plot(ks, u_score, marker="o", color="#58a6ff", label="Quality (U)")
        ax_ms.plot(ks, h_score, marker="s", color="#2ca02c", linestyle="--", label="Homogeneity (H)")
        ax_ms.plot(ks, c_score, marker="^", color="#ff7f0e", linestyle="--", label="Completeness (C)")
        ax_ms.plot(ks, q_score, marker="x", color="#d62728", linestyle=":", label="Modularity (Q)")
        ax_ms.axvline(best_k, color="#d62728", linestyle="-.", label=f"Selected k={best_k}")
        
        ax_ms.set_xlabel("Number of Clusters (k)", color="white")
        ax_ms.set_ylabel("Normalized Score", color="white")
        ax_ms.set_xticks(ks)
        ax_ms.tick_params(colors="white")
        ax_ms.legend()
        fig_ms.tight_layout()
        st.pyplot(fig_ms)

    # Heatmaps and scalp networks
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Consensus Co-occurrence Matrix")
        fig_heat = plot_connectivity_heatmap(consensus_matrix, consensus_labels, sort_heatmap, channel_names)
        st.pyplot(fig_heat)
    with col_b:
        st.markdown("#### Scalp Consensus Network Map")
        fig_net, A_thr = plot_paper_electrode_network(mean_adj, consensus_labels, max_plot_edges, show_labels, xy, channel_names)
        st.pyplot(fig_net)

    # Node table
    st.markdown("#### Consensus Community assignments & Sensor Connectivity Stats")
    metric_table = node_metric_table(mean_adj, consensus_labels, A_thr, channel_names)
    st.dataframe(
        metric_table.sort_values(["Community", "Thresholded Strength"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True
    )





if __name__ == "__main__":
    main()
