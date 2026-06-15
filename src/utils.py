import numpy as np
import scipy.signal
import mne
from pathlib import Path
from typing import List
from fcca.fcca_by_method import run_fcca_model_selection, Config as FCCAConfig
from preprocessing.config_1024 import config as prep_config

def compute_plv_tensors(sub_files: List[str], config, use_fallback: bool = True) -> tuple:
    """
    Compute PLV connectivity tensors from theta-balanced epochs.
    
    Returns:
        tensor_correct: 4D array of shape (subjects, nodes, nodes, time)
        tensor_incorrect: 4D array of shape (subjects, nodes, nodes, time)
    """
    sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
    n_channels = len(sample_epochs.ch_names)
    n_points = sample_epochs.get_data(verbose=False).shape[2]
    
    n_subjects = len(sub_files)
    tensor_correct = np.zeros((n_subjects, n_channels, n_channels, n_points), dtype=np.float32)
    tensor_incorrect = np.zeros((n_subjects, n_channels, n_channels, n_points), dtype=np.float32)
    
    print(f"Computing PLV Tensors for {n_subjects} subjects...")
    
    for sub_idx, f in enumerate(sub_files):
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        sfreq = epochs.info['sfreq']
        
        for condition, storage in [('Correct', tensor_correct), ('Incorrect', tensor_incorrect)]:
            # Get data shape: (trials, channels, time_points)
            cond_epochs = epochs[condition]
            if len(cond_epochs) == 0:
                continue
                
            data = cond_epochs.get_data(verbose=False)
            n_trials = data.shape[0]
            
            # 1. Filter in the theta band (4-8 Hz) using MNE's filter
            filtered_data = mne.filter.filter_data(
                data.astype(np.float64), 
                sfreq, 
                l_freq=config.proc.THETA_BAND[0], 
                h_freq=config.proc.THETA_BAND[1], 
                verbose=False
            )
            
            # 2. Compute analytic signal using Hilbert transform
            analytic_signal = scipy.signal.hilbert(filtered_data, axis=-1)
            
            # 3. Extract phase
            phases = np.angle(analytic_signal)
            
            # 4. Compute PLV: |mean_trials( exp(j * (phi_c1 - phi_c2)) )|
            # Vectorized implementation:
            phases_exp = np.exp(1j * phases)
            cross_spec = np.einsum('tcp,tdp->cdp', phases_exp, np.conj(phases_exp)) / n_trials
            plv = np.abs(cross_spec).astype(np.float32)
            
            # Symmetrize and zero-out diagonal
            for p in range(n_points):
                plv_p = plv[:, :, p]
                plv_p = 0.5 * (plv_p + plv_p.T)
                np.fill_diagonal(plv_p, 0)
                plv[:, :, p] = plv_p
                
            storage[sub_idx] = plv
            
        print(f"  [{sub_idx + 1}/{n_subjects}] Completed {Path(f).name}")
        
    return tensor_correct, tensor_incorrect

def graph_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Extract subject-averaged upper-triangle edge weights over time."""
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {X.shape}")
    _, n_nodes, n_nodes_2, _ = X.shape
    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square connectivity matrices, got shape {X.shape}")

    upper = np.triu_indices(n_nodes, k=1)
    features = X[:, upper[0], upper[1], :].mean(axis=0).T
    features = np.asarray(features, dtype=float)
    features[~np.isfinite(features)] = 0

    std = features.std(axis=0)
    valid = std > 0
    features[:, valid] = (features[:, valid] - features[:, valid].mean(axis=0)) / std[valid]
    features[:, ~valid] = 0
    return features

def segment_sse(prefix_sum: np.ndarray, prefix_sq_sum: np.ndarray, start: int, end: int) -> float:
    n_samples = end - start
    if n_samples <= 0:
        return np.inf
    segment_sum = prefix_sum[end] - prefix_sum[start]
    segment_sq_sum = prefix_sq_sum[end] - prefix_sq_sum[start]
    return float(segment_sq_sum.sum() - np.square(segment_sum).sum() / n_samples)

def detect_change_points_exact(features: np.ndarray, n_bkps: int = 2, min_size: int = 10) -> list:
    """Exact dynamic programming change-point detector using squared-error segments."""
    n_times = features.shape[0]
    n_segments = n_bkps + 1
    min_size = max(1, min(min_size, n_times // n_segments))

    if n_times < n_segments:
        return []

    prefix_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features, axis=0)])
    prefix_sq_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features * features, axis=0)])

    dp = np.full((n_segments + 1, n_times + 1), np.inf)
    previous = np.full((n_segments + 1, n_times + 1), -1, dtype=int)
    dp[0, 0] = 0

    for segment_idx in range(1, n_segments + 1):
        min_end = segment_idx * min_size
        max_end = n_times - (n_segments - segment_idx) * min_size
        for end in range(min_end, max_end + 1):
            best_cost = np.inf
            best_start = -1
            min_start = (segment_idx - 1) * min_size
            max_start = end - min_size
            for start in range(min_start, max_start + 1):
                cost = dp[segment_idx - 1, start] + segment_sse(prefix_sum, prefix_sq_sum, start, end)
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[segment_idx, end] = best_cost
            previous[segment_idx, end] = best_start

    if not np.isfinite(dp[n_segments, n_times]):
        return []

    boundaries = []
    end = n_times
    for segment_idx in range(n_segments, 0, -1):
        start = previous[segment_idx, end]
        if start <= 0:
            break
        boundaries.append(start)
        end = start

    return sorted(boundaries)

def make_intervals_from_change_points(n_times: int, change_points: list, names=("pre_ern", "ern", "post_ern")) -> dict:
    """Group frames into pre, event, post intervals based on change points."""
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

def run_fcca_for_interval(tensor_4d: np.ndarray, frames: np.ndarray, k_range=None):
    """Run FCCA consensus clustering on subject networks for a specific frame interval."""
    if k_range is None:
        k_range = prep_config.algo.FCCA_K_RANGE
    n_subjects, n_nodes, _, _ = tensor_4d.shape
    graphs = np.zeros((n_nodes, n_nodes, n_subjects), dtype=np.float32)
    
    for s in range(n_subjects):
        W = np.mean(np.take(tensor_4d[s], frames, axis=-1), axis=-1)
        # Symmetrize, positive only, zero diagonal
        W = (W + W.T) / 2.0
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        graphs[:, :, s] = W.astype(np.float32)
        
    cfg = FCCAConfig(kRange=tuple(k_range), minCommunitySize=prep_config.algo.FCCA_MIN_COMMUNITY_SIZE)
    fcca_res = run_fcca_model_selection(graphs, list(k_range), cfg)
    mean_adjacency = np.mean(graphs, axis=2)

    return fcca_res["cooccurrence"], fcca_res["labels"], float(fcca_res["bestModularity"]), mean_adjacency
