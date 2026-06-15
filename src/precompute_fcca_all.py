import os
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Ensure project dirs are in Python path at module level (crucial for spawned processes on Windows)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))
if str(ROOT / "src/low_rank_extraction") not in sys.path:
    sys.path.append(str(ROOT / "src/low_rank_extraction"))
if str(ROOT / "notebooks") not in sys.path:
    sys.path.append(str(ROOT / "notebooks"))

from scipy.linalg import eigh
from HoSVD import hosvd_low_rank

try:
    import networkx as nx
    if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
        nx.from_numpy_matrix = nx.from_numpy_array
except ImportError:
    nx = None

def sanitize_adjacency(A):
    A = np.asarray(A, dtype=float).copy()
    A[~np.isfinite(A)] = 0
    A = (A + A.T) / 2
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    return A

def fiedler_split(A):
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]
    if n_nodes < 2 or np.allclose(A, 0):
        return np.zeros(n_nodes, dtype=int)

    L = np.diag(A.sum(axis=1)) - A
    _, eigenvectors = eigh(L)
    fvec = eigenvectors[:, 1]
    labels = (fvec > 0).astype(int)

    if len(np.unique(labels)) < 2:
        labels = (fvec > np.median(fvec)).astype(int)

    if len(np.unique(labels)) < 2:
        order = np.argsort(fvec)
        labels = np.zeros(n_nodes, dtype=int)
        labels[order[n_nodes // 2 :]] = 1

    return labels

def labels_to_communities(labels):
    labels = np.asarray(labels, dtype=int)
    return [
        set(np.flatnonzero(labels == label).tolist())
        for label in sorted(np.unique(labels))
    ]

def partition_to_labels(partition, n_nodes):
    labels = np.zeros(n_nodes, dtype=int)
    ordered = sorted((sorted(group) for group in partition if group), key=lambda group: group[0])
    for label, group in enumerate(ordered):
        labels[np.asarray(group, dtype=int)] = label
    return labels

def weighted_modularity(A, labels):
    A = sanitize_adjacency(A)
    labels = np.asarray(labels, dtype=int)
    total_weight = A.sum() / 2.0
    if total_weight <= 0:
        return np.nan

    degrees = A.sum(axis=1)
    score = 0.0
    for label in np.unique(labels):
        nodes = np.flatnonzero(labels == label)
        internal_weight = A[np.ix_(nodes, nodes)].sum() / 2.0
        degree_sum = degrees[nodes].sum()
        score += internal_weight / total_weight - (degree_sum / (2.0 * total_weight)) ** 2
    return float(score)

def recursive_fiedler_communities(A, min_size=3, min_gain=1e-4):
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]
    partition = [list(range(n_nodes))]

    changed = True
    while changed:
        changed = False
        best_gain = min_gain
        best_index = None
        best_split = None

        for idx, nodes in enumerate(partition):
            if len(nodes) < 2 * min_size:
                continue

            subA = A[np.ix_(nodes, nodes)]
            sublabels = fiedler_split(subA)
            if len(np.unique(sublabels)) < 2:
                continue

            left = [nodes[i] for i in np.flatnonzero(sublabels == 0)]
            right = [nodes[i] for i in np.flatnonzero(sublabels == 1)]
            if len(left) < min_size or len(right) < min_size:
                continue

            gain = weighted_modularity(subA, sublabels)
            if np.isfinite(gain) and gain > best_gain:
                best_gain = gain
                best_index = idx
                best_split = [left, right]

        if best_index is not None and best_split is not None:
            partition = partition[:best_index] + best_split + partition[best_index + 1 :]
            changed = True

    labels = partition_to_labels(partition, n_nodes)
    return labels, weighted_modularity(A, labels), "Adaptive recursive Fiedler"

def communities_to_labels(communities, n_nodes):
    labels = np.zeros(n_nodes, dtype=int)
    for community_idx, community in enumerate(communities):
        for node in community:
            labels[node] = community_idx
    return labels

def clustering_profile_for_interval(interval_name=None):
    CORE_INTERVAL_NAMES = {"ern", "crn"}
    COMPACT_INTERVAL_NAMES = {"pre_ern", "post_ern", "pre_crn", "post_crn"}
    if interval_name in CORE_INTERVAL_NAMES:
        return "detailed"
    if interval_name in COMPACT_INTERVAL_NAMES:
        return "compact"
    return "balanced"

def modularity_communities(A, seed=42, clustering_profile="balanced"):
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]

    if n_nodes < 2 or np.allclose(A, 0):
        return np.zeros(n_nodes, dtype=int), np.nan, "empty graph"

    if clustering_profile == "detailed":
        recursive_tolerance = 0.08
        recursive_min_size = 3
    elif clustering_profile == "compact":
        recursive_tolerance = 0.0
        recursive_min_size = 6
    else:
        recursive_tolerance = 0.06
        recursive_min_size = 3

    recursive_labels, recursive_score, recursive_method = recursive_fiedler_communities(
        A,
        min_size=recursive_min_size,
    )

    if nx is None:
        return recursive_labels, recursive_score, recursive_method

    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return np.zeros(n_nodes, dtype=int), np.nan, "empty graph"

    candidates = [(recursive_labels, recursive_score, recursive_method)]

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
        method = "Louvain modularity"
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        method = "Greedy modularity"

    communities = [set(community) for community in communities if len(community) > 0]
    labels = communities_to_labels(communities, n_nodes)

    try:
        score = nx.community.modularity(G, communities, weight="weight")
    except Exception:
        score = np.nan
    candidates.append((labels, float(score), method))

    finite_candidates = [
        candidate for candidate in candidates if np.isfinite(candidate[1])
    ]
    if not finite_candidates:
        return recursive_labels, recursive_score, recursive_method

    best_labels, best_score, best_method = max(finite_candidates, key=lambda item: item[1])
    recursive_k = len(np.unique(recursive_labels))
    best_k = len(np.unique(best_labels))
    if recursive_k > best_k and recursive_score >= best_score - recursive_tolerance:
        return recursive_labels, float(recursive_score), recursive_method

    return best_labels, float(best_score), best_method

def build_consensus_matrix(labels_all, n_nodes):
    W = np.zeros((n_nodes, n_nodes), dtype=float)

    for labels in labels_all:
        same = labels[:, None] == labels[None, :]
        W += same.astype(float)

    W /= max(len(labels_all), 1)
    return W

def make_intervals_from_change_points(n_times, change_points, names=("pre_ern", "ern", "post_ern")):
    boundaries = [0]
    boundaries.extend(int(point) for point in sorted(change_points) if 0 < point < n_times)
    boundaries.append(n_times)

    if len(boundaries) == 6:
        return {
            "pre_ern": np.arange(boundaries[1], boundaries[2]),
            "ern": np.arange(boundaries[2], boundaries[3]),
            "post_ern": np.arange(boundaries[3], boundaries[4]),
        }

    if len(boundaries) != len(names) + 1:
        splits = np.array_split(np.arange(n_times), len(names))
        return {
            name: frames
            for name, frames in zip(names, splits)
            if len(frames) > 0
        }

    return {
        name: np.arange(start, end)
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:])
        if end > start
    }

TENSOR_FILES = {
    "incorrect": ROOT / "outputs/tensor_4d_1024/tensor_incorrect_4d.npy",
    "correct":   ROOT / "outputs/tensor_4d_1024/tensor_correct_4d.npy",
}

TENSOR_FILES_FALLBACK = {
    "incorrect": ROOT / "outputs/processed_1024/03_connectivity_tensors/tensor_incorrect_4d.npy",
    "correct":   ROOT / "outputs/processed_1024/03_connectivity_tensors/tensor_correct_4d.npy",
}

CP_ALGO_FILES = {
    "HOSVD":       ROOT / "outputs/fcca_results/HOSVD/HOSVD_ChangePoints.npy",
    "HO-RLSL":     ROOT / "outputs/fcca_results/HO_RLSL/HORLSL_ChangePoints.npy",
    "PELT":        ROOT / "outputs/fcca_results/PELT/PELT_ChangePoints.npy",
    "DMD":         ROOT / "outputs/fcca_results/DMD/DMD_ChangePoints.npy",
    "CP-Tracking": ROOT / "outputs/fcca_results/CP_Tracking/CP_ChangePoints.npy",
}

HO_RLSL_LOWRANK_FILES = {
    "incorrect": ROOT / "outputs/fcca_results/HO_RLSL/HORLSL_1024Hz_ERN_LowRank.npy",
    "correct":   ROOT / "outputs/fcca_results/HO_RLSL/HORLSL_1024Hz_CRN_LowRank.npy",
}

_CP_KEY_MAP = {"incorrect": "1024Hz_ERN", "correct": "1024Hz_CRN"}
INTERVAL_NAMES = ("pre_ern", "ern", "post_ern")

def process_single_subject(args):
    """Worker function to process all time windows for a single subject."""
    s, X_clean_subject, n_times, n_nodes = args
    
    labels_s = np.zeros((n_times, n_nodes), dtype=int)
    scores_s = np.zeros(n_times, dtype=float)
    method_str = "unknown"

    for t in range(n_times):
        A = sanitize_adjacency(X_clean_subject[:, :, t])
        labels, score, method = modularity_communities(A)
        labels_s[t, :] = labels
        scores_s[t] = score
        method_str = method

    return s, labels_s, scores_s, method_str

def main():
    print("=" * 80)
    print("PRE-COMPUTING FCCA RESULTS FOR ALL ALGORITHMS AND CONDITIONS (PARALLELIZED)")
    print("=" * 80)

    precomputed_data = {}

    for condition in ["incorrect", "correct"]:
        print(f"\n>>> Processing condition: {condition.upper()}...")
        raw_tensor_path = TENSOR_FILES[condition]
        if not raw_tensor_path.exists():
            raw_tensor_path = TENSOR_FILES_FALLBACK[condition]
        if not raw_tensor_path.exists():
            print(f"Error: Raw tensor not found at {TENSOR_FILES[condition]} or fallback {TENSOR_FILES_FALLBACK[condition]}")
            continue

        print(f"  Loading raw tensor: {raw_tensor_path.name}...")
        X_raw = np.load(raw_tensor_path)
        n_subjects, n_nodes, _, n_times = X_raw.shape
        times_ms = np.linspace(-1000.0, 1000.0, n_times)

        # Precompute HoSVD low-rank
        print("  Computing HoSVD low-rank reconstruction...")
        X_hosvd = hosvd_low_rank(X_raw, ranks=(10, 10, 10))

        # Load HO-RLSL low-rank
        ho_rlsl_lowrank_path = HO_RLSL_LOWRANK_FILES[condition]
        X_ho_rlsl = None
        if ho_rlsl_lowrank_path.exists():
            print(f"  Loading pre-computed HO-RLSL low-rank tensor: {ho_rlsl_lowrank_path.name}...")
            X_ho_rlsl = np.load(ho_rlsl_lowrank_path)
            if X_ho_rlsl.shape[0] != n_subjects:
                print(f"    Transposing HO-RLSL tensor shape {X_ho_rlsl.shape} to subject-first...")
                X_ho_rlsl = np.transpose(X_ho_rlsl, (2, 0, 1, 3))
        else:
            print(f"  [WARN] HO-RLSL low-rank tensor not found. Falling back to HoSVD.")
            X_ho_rlsl = X_hosvd

        # Precompute and cache subject-level communities for both low-rank methods using ProcessPoolExecutor
        labels_cache = {}
        scores_cache = {}
        methods_cache = {}

        for lr_name, X_clean in [("HOSVD", X_hosvd), ("HO-RLSL", X_ho_rlsl)]:
            print(f"  Precomputing subject labels for {lr_name} sequentially...")
            labels_arr = np.zeros((n_subjects, n_times, n_nodes), dtype=int)
            scores_arr = np.zeros((n_subjects, n_times), dtype=float)
            method_str = "unknown"

            for s in range(n_subjects):
                _, labels_s, scores_s, method_s = process_single_subject((s, X_clean[s], n_times, n_nodes))
                labels_arr[s, :, :] = labels_s
                scores_arr[s, :] = scores_s
                method_str = method_s

            labels_cache[lr_name] = labels_arr
            scores_cache[lr_name] = scores_arr
            methods_cache[lr_name] = method_str
            print(f"    Completed subject-level clustering cache for {lr_name}.")

        # Now compute FCCA for each change-point algorithm in seconds
        for algo in ["HOSVD", "HO-RLSL", "PELT", "DMD", "CP-Tracking"]:
            print(f"\n  Generating FCCA consensus networks for algorithm: {algo}...")
            cp_file = CP_ALGO_FILES[algo]
            if not cp_file.exists():
                print(f"    [WARN] Change point file not found for {algo} at {cp_file.name}. Skipping.")
                continue

            if algo == "HO-RLSL":
                if condition == "incorrect":
                    ms_values = [-860.4, -47.9, 139.6, 577.1, 702.1]
                else:
                    ms_values = [-860.0, -47.9, 139.6, 577.1]
                frames = [int(np.argmin(np.abs(times_ms - ms))) for ms in ms_values]
                intervals = {
                    "pre_ern": np.arange(frames[0], frames[1]),
                    "ern": np.arange(frames[1], frames[2]),
                    "post_ern": np.arange(frames[2], frames[3]),
                }
            else:
                cp_dict = np.load(str(cp_file), allow_pickle=True).item()
                cond_key = _CP_KEY_MAP[condition]
                if cond_key not in cp_dict:
                    print(f"    [WARN] Key {cond_key} not in {algo} change points. Skipping.")
                    continue

                ms_values = cp_dict[cond_key]
                frames = []
                for ms in ms_values:
                    idx = int(np.argmin(np.abs(times_ms - ms)))
                    if 0 < idx < n_times:
                        frames.append(idx)
                frames = sorted(set(frames))

                intervals = make_intervals_from_change_points(n_times, frames)
            
            # Select correct Clean Tensor and cached labels
            lr_name = "HO-RLSL" if algo == "HO-RLSL" else "HOSVD"
            X_clean = X_ho_rlsl if algo == "HO-RLSL" else X_hosvd
            cached_labels = labels_cache[lr_name]
            cached_scores = scores_cache[lr_name]
            cached_method = methods_cache[lr_name]
 
            for name, interval_frames in intervals.items():
                print(f"    -> Consensus clustering interval '{name}' (frames {interval_frames[0]}-{interval_frames[-1]})...")
                
                # Slice graphs and labels
                sliced_labels = cached_labels[:, interval_frames, :]  # shape: (subjects, frames_in_interval, nodes)
                reshaped_labels = sliced_labels.reshape(-1, n_nodes)  # shape: (subjects * frames_in_interval, nodes)
                
                # Build consensus matrix
                consensus_matrix = build_consensus_matrix(reshaped_labels, n_nodes)
                
                # mod cluster consensus matrix
                clustering_profile = clustering_profile_for_interval(name)
                consensus_labels, consensus_modularity, consensus_method = modularity_communities(
                    consensus_matrix,
                    clustering_profile=clustering_profile,
                )
                
                # Mean adjacency (symmetrized, zero diagonal)
                mean_adjacency = X_clean[:, :, :, interval_frames].mean(axis=(0, 3))
                mean_adjacency = sanitize_adjacency(mean_adjacency)

                # Store precomputed keys
                precomputed_data[f"{condition}_{algo}_{name}_consensus_matrix"] = consensus_matrix
                precomputed_data[f"{condition}_{algo}_{name}_consensus_labels"] = consensus_labels
                precomputed_data[f"{condition}_{algo}_{name}_consensus_modularity"] = np.asarray(consensus_modularity)
                precomputed_data[f"{condition}_{algo}_{name}_community_method"] = np.asarray(
                    f"{consensus_method or cached_method} ({clustering_profile})"
                )
                precomputed_data[f"{condition}_{algo}_{name}_mean_adjacency"] = mean_adjacency

    # Save all precomputed data
    out_file = ROOT / "outputs/fcca_results/precomputed_fcca_all.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **precomputed_data)
    print("\n" + "=" * 80)
    print(f"SUCCESS: Saved all precomputed FCCA results to:\n  {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
