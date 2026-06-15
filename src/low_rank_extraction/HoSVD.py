from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class HoSVDChangePointResult:
    change_points: list[int]
    method: str
    ranks: tuple[int | None, int | None, int | None]
    original_shape: tuple[int, int, int, int]
    feature_shape: tuple[int, int]
    min_size: int
    n_bkps: int


def validate_subject_first_tensor(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape (subjects, nodes, nodes, time), got {tensor.shape}")
    if tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"Expected square node-node matrices, got {tensor.shape}")
    return tensor


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    tensor = np.asarray(tensor)
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    result = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(result, 0, mode)


def parse_rank_tuple(value: str | None) -> tuple[int | None, int | None, int | None]:
    if value is None or value.strip() == "":
        return (10, 10, 10)
    entries: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"none", "null", "-"}:
            entries.append(None)
        else:
            entries.append(int(item))
    if len(entries) != 3:
        raise ValueError("Expected three ranks for subject,node,node modes, e.g. '10,10,10'")
    return entries[0], entries[1], entries[2]


def hosvd_bases(
    tensor: np.ndarray,
    ranks: tuple[int | None, int | None, int | None] = (10, 10, 10),
    modes: tuple[int, int, int] = (0, 1, 2),
) -> list[np.ndarray]:
    tensor = validate_subject_first_tensor(tensor).astype(float, copy=False)
    bases: list[np.ndarray] = []
    for mode, rank in zip(modes, ranks):
        data = unfold(tensor, mode)
        data = data / np.sqrt(max(data.shape[1], 1))
        u, _, _ = np.linalg.svd(data, full_matrices=False)
        if rank is None:
            keep = u.shape[1]
        else:
            keep = max(1, min(int(rank), u.shape[1]))
        bases.append(u[:, :keep])
    return bases


def hosvd_low_rank(
    tensor: np.ndarray,
    ranks: tuple[int | None, int | None, int | None] = (10, 10, 10),
    tie_symmetric_node_modes: bool = True,
) -> np.ndarray:
    """Project a subject-first connectivity tensor onto HoSVD subspaces.

    The time mode is intentionally not compressed; change-point detection needs
    to preserve the temporal axis. This is a HoSVD-style denoising baseline for
    `(subjects, nodes, nodes, time)` connectivity tensors.
    """
    tensor = validate_subject_first_tensor(tensor).astype(float, copy=False)
    bases = hosvd_bases(tensor, ranks=ranks)
    if tie_symmetric_node_modes and bases[1].shape[0] == bases[2].shape[0]:
        bases[2] = bases[1].copy()

    low_rank = tensor
    for mode, basis in enumerate(bases):
        low_rank = mode_dot(low_rank, basis.T, mode)
    for mode, basis in enumerate(bases):
        low_rank = mode_dot(low_rank, basis, mode)

    low_rank = np.asarray(low_rank, dtype=float)
    low_rank = 0.5 * (low_rank + np.swapaxes(low_rank, 1, 2))
    diagonal = np.arange(low_rank.shape[1])
    low_rank[:, diagonal, diagonal, :] = 0.0
    return low_rank


def graph_feature_matrix(tensor: np.ndarray) -> np.ndarray:
    tensor = validate_subject_first_tensor(tensor)
    _, n_nodes, _, _ = tensor.shape
    upper = np.triu_indices(n_nodes, k=1)
    features = tensor[:, upper[0], upper[1], :].mean(axis=0).T
    features = np.asarray(features, dtype=float)
    features[~np.isfinite(features)] = 0.0

    std = features.std(axis=0)
    valid = std > 0
    features[:, valid] = (features[:, valid] - features[:, valid].mean(axis=0)) / std[valid]
    features[:, ~valid] = 0.0
    return features


def segment_sse(prefix_sum: np.ndarray, prefix_sq_sum: np.ndarray, start: int, end: int) -> float:
    n_samples = end - start
    if n_samples <= 0:
        return np.inf
    segment_sum = prefix_sum[end] - prefix_sum[start]
    segment_sq_sum = prefix_sq_sum[end] - prefix_sq_sum[start]
    return float(segment_sq_sum.sum() - np.square(segment_sum).sum() / n_samples)


def detect_change_points_exact(features: np.ndarray, n_bkps: int = 2, min_size: int = 10) -> list[int]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"Expected feature matrix shape (time, features), got {features.shape}")

    n_times = features.shape[0]
    n_segments = int(n_bkps) + 1
    min_size = max(1, min(int(min_size), n_times // n_segments))
    if n_times < n_segments:
        return []

    prefix_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features, axis=0)])
    prefix_sq_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features * features, axis=0)])
    dp = np.full((n_segments + 1, n_times + 1), np.inf)
    previous = np.full((n_segments + 1, n_times + 1), -1, dtype=int)
    dp[0, 0] = 0.0

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

    boundaries: list[int] = []
    end = n_times
    for segment_idx in range(n_segments, 0, -1):
        start = int(previous[segment_idx, end])
        if start <= 0:
            break
        boundaries.append(start)
        end = start
    return sorted(boundaries)


def find_change_points(
    tensor: np.ndarray,
    ranks: tuple[int | None, int | None, int | None] = (10, 10, 10),
    n_bkps: int = 2,
    min_size: int = 10,
    use_low_rank: bool = True,
) -> HoSVDChangePointResult:
    tensor = validate_subject_first_tensor(tensor)
    if use_low_rank:
        detector_tensor = hosvd_low_rank(tensor, ranks=ranks)
        method = "HoSVD low-rank + exact SSE"
    else:
        detector_tensor = tensor
        method = "raw tensor + exact SSE"

    features = graph_feature_matrix(detector_tensor)
    change_points = detect_change_points_exact(features, n_bkps=n_bkps, min_size=min_size)
    return HoSVDChangePointResult(
        change_points=change_points,
        method=method,
        ranks=tuple(ranks),
        original_shape=tuple(int(value) for value in tensor.shape),
        feature_shape=tuple(int(value) for value in features.shape),
        min_size=int(min_size),
        n_bkps=int(n_bkps),
    )


def find_change_points_from_file(
    input_path: Path,
    ranks: tuple[int | None, int | None, int | None] = (10, 10, 10),
    n_bkps: int = 2,
    min_size: int = 10,
    use_low_rank: bool = True,
) -> HoSVDChangePointResult:
    tensor = np.load(input_path)
    return find_change_points(
        tensor,
        ranks=ranks,
        n_bkps=n_bkps,
        min_size=min_size,
        use_low_rank=use_low_rank,
    )


def save_result(result: HoSVDChangePointResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        change_points=np.asarray(result.change_points, dtype=int),
        method=np.asarray(result.method),
        ranks=np.asarray([-1 if rank is None else rank for rank in result.ranks], dtype=int),
        original_shape=np.asarray(result.original_shape, dtype=int),
        feature_shape=np.asarray(result.feature_shape, dtype=int),
        min_size=np.asarray(result.min_size, dtype=int),
        n_bkps=np.asarray(result.n_bkps, dtype=int),
        config=np.asarray(asdict(result), dtype=object),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find change points with a HoSVD low-rank baseline.")
    parser.add_argument("--input", type=Path, default=Path("tensor_4d/tensor_incorrect_4d.npy"))
    parser.add_argument("--output", type=Path, default=Path("fcca_results/hosvd_change_points.npz"))
    parser.add_argument("--ranks", default="10,10,10", help="Subject,node,node ranks, e.g. 10,10,10")
    parser.add_argument("--n-bkps", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip HoSVD projection and detect change points on the raw tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = parse_rank_tuple(args.ranks)
    result = find_change_points_from_file(
        args.input,
        ranks=ranks,
        n_bkps=args.n_bkps,
        min_size=args.min_size,
        use_low_rank=not args.raw,
    )
    save_result(result, args.output)
    print(f"Saved {args.output}")
    print(f"Method: {result.method}")
    print(f"Change points: {result.change_points}")


if __name__ == "__main__":
    main()
