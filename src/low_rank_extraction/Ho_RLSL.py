from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


SUBJECT_FIRST_LAYOUT = "subjects,nodes,nodes,time"
PAPER_TIME_LAST_LAYOUT = "nodes,nodes,subjects,time"


def validate_subject_first_tensor(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape (subjects, nodes, nodes, time), got {tensor.shape}")
    if tensor.shape[1] != tensor.shape[2]:
        raise ValueError(
            "Expected square node-node connectivity matrices in axes 1 and 2, "
            f"got shape {tensor.shape}"
        )
    return tensor


def validate_paper_time_last_tensor(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence)
    if sequence.ndim != 4:
        raise ValueError(f"Expected tensor shape (nodes, nodes, subjects, time), got {sequence.shape}")
    if sequence.shape[0] != sequence.shape[1]:
        raise ValueError(
            "Expected square node-node connectivity matrices in axes 0 and 1, "
            f"got shape {sequence.shape}"
        )
    return sequence


def subject_first_to_paper_time_last(tensor: np.ndarray) -> np.ndarray:
    """Convert `(subjects, nodes, nodes, time)` to `(nodes, nodes, subjects, time)`."""
    tensor = validate_subject_first_tensor(tensor)
    return np.transpose(tensor, (1, 2, 0, 3))


def paper_time_last_to_subject_first(sequence: np.ndarray) -> np.ndarray:
    """Convert `(nodes, nodes, subjects, time)` to `(subjects, nodes, nodes, time)`."""
    sequence = validate_paper_time_last_tensor(sequence)
    return np.transpose(sequence, (2, 0, 1, 3))


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Return mode-n unfolding with mode fibers as columns."""
    tensor = np.asarray(tensor)
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Compute tensor x_mode matrix."""
    result = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(result, 0, mode)


def multi_mode_dot(
    tensor: np.ndarray,
    matrices: Iterable[np.ndarray],
    modes: Iterable[int] | None = None,
) -> np.ndarray:
    result = np.asarray(tensor)
    matrices = tuple(matrices)
    if modes is None:
        modes = range(len(matrices))
    for matrix, mode in zip(matrices, modes):
        result = mode_dot(result, matrix, int(mode))
    return result


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.reshape(matrix.shape[0], 0)
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > np.finfo(float).eps
    return q[:, keep]


def _empty_basis(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, 0), dtype=float)


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Projection Frobenius distance normalized to roughly [0, sqrt(2)]."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("Subspaces must have the same ambient dimension")
    if left.shape[1] == 0 and right.shape[1] == 0:
        return 0.0
    left_projection = left @ left.T if left.shape[1] else np.zeros((left.shape[0], left.shape[0]))
    right_projection = right @ right.T if right.shape[1] else np.zeros((right.shape[0], right.shape[0]))
    scale = np.sqrt(max(left.shape[1], right.shape[1], 1))
    return float(np.linalg.norm(left_projection - right_projection, ord="fro") / scale)


def parse_optional_int_tuple(value: str | None) -> tuple[int | None, ...] | None:
    if value is None or value.strip() == "":
        return None
    entries: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"none", "null", "-"}:
            entries.append(None)
        else:
            entries.append(int(item))
    return tuple(entries)


def parse_int_pair(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip().lower() in {"none", "null", "-"}:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected an integer pair like '0,1'")
    return parts[0], parts[1]


@dataclass
class HoRLSLConfig:
    """Configuration for Higher-Order Recursive Low-Rank + Sparse Learning.

    This implementation follows the Ho-RLSL structure from Ozdemir et al.:
    initialize Tucker-mode subspaces, remove sparse outliers, recursively
    update subspace directions, and mark updates as change points.

    The paper recovers the sparse tensor from the orthogonally projected
    measurement with GTCS-S. This module uses a practical OMP-style sparse
    tensor pursuit by default, with FISTA and proxy options for experiments.
    """

    train_length: int = 80
    alpha: int = 64
    sigma_min: float = 0.11
    init_method: str = "hosvd"
    threshold_mode: str = "relative"
    normalize_unfoldings: bool = True
    sparse_solver: str = "gtcs_s_omp"
    lambda_sparse: float | None = None
    epsilon: float | None = None
    epsilon_mode: str = "absolute"
    fista_max_iter: int = 20
    fista_tol: float = 1e-4
    fista_step_size: float = 1.0
    sparsity: int = 8
    residual_tol: float = 1e-3
    max_atoms: int | None = None
    sparse_threshold: float = 3.0
    sparse_threshold_mode: str = "mad"
    enforce_symmetric_connectivity: bool = True
    zero_diagonal: bool = True
    max_ranks: tuple[int | None, ...] | None = (15, 15, 10)
    tie_symmetric_modes: tuple[int, int] | None = (0, 1)
    use_paper_change_rule: bool = True
    change_point_position: str = "midpoint"
    change_point_modes: tuple[int, ...] | None = None
    subspace_change_threshold: float = 0.15
    min_cp_distance_ms: float = 50.0
    score_smoothing_ms: float = 25.0
    threshold_method: str = "mad"
    threshold_k: float = 3.0
    mode_weights: tuple[float, ...] = (1.0, 1.0, 0.4)
    sampling_rate: float | None = 1024.0
    target_window_ms: tuple[float, float] | None = (-150.0, 150.0)
    target_anchor_ms: float = 50.0
    store_rank_history: bool = True


@dataclass
class HoRLSLResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    change_points: list[int]
    final_bases: list[np.ndarray]
    rank_history: list[tuple[int, tuple[int, ...]]]
    update_times: list[int]
    change_history: list[dict[str, int]]
    raw_change_points: list[int]
    filtered_change_points: list[int]
    change_point_ms: np.ndarray
    change_point_modes: list[int]
    change_scores: np.ndarray
    change_score_times: np.ndarray
    change_point_score_components: np.ndarray
    solver_params: dict[str, object]
    input_layout: str
    output_layout: str
    internal_layout: str
    original_shape: tuple[int, ...]
    internal_shape: tuple[int, ...]
    initial_ranks: tuple[int, ...]
    initial_thresholds: tuple[float, ...]
    config: dict[str, object]


class HoRLSL:
    """Higher-order recursive low-rank + sparse tracker for time-last tensors.

    The paper-like internal layout is `(nodes, nodes, subjects, time)`. For
    this repo's saved connectivity tensors, use `fit_transform_subject_first`,
    because those arrays are `(subjects, nodes, nodes, time)`.
    """

    def __init__(self, config: HoRLSLConfig | None = None):
        self.config = config or HoRLSLConfig()
        self.bases_: list[np.ndarray] | None = None
        self.thresholds_: list[float] | None = None
        self.initial_core_: np.ndarray | None = None
        self.initial_full_factors_: list[np.ndarray] | None = None
        self.initial_time_basis_: np.ndarray | None = None
        self.initial_singular_values_: list[np.ndarray] | None = None
        self.initial_ranks_: tuple[int, ...] | None = None

    def fit_transform(self, sequence: np.ndarray) -> HoRLSLResult:
        sequence = validate_paper_time_last_tensor(sequence).astype(float, copy=False)

        n_modes = sequence.ndim - 1
        n_times = sequence.shape[-1]
        if n_times <= self.config.train_length:
            raise ValueError("train_length must be smaller than the number of time samples")

        self._validate_config(n_modes)
        train = sequence[..., : self.config.train_length]
        self.bases_, self.thresholds_ = self._initial_subspaces(train)
        self._apply_tied_modes()
        self.initial_ranks_ = tuple(base.shape[1] for base in self.bases_)

        low_rank = np.zeros_like(sequence, dtype=float)
        sparse = np.zeros_like(sequence, dtype=float)
        rank_history: list[tuple[int, tuple[int, ...]]] = []
        change_points: list[int] = []
        raw_change_points: list[int] = []
        update_times: list[int] = []
        change_history: list[dict[str, int]] = []
        change_score_times: list[int] = []
        change_score_rows: list[dict[str, float]] = []
        update_buffer: list[np.ndarray] = []
        diagnostic_buffer: list[dict[str, object]] = []

        for t in range(n_times):
            m_t = sequence[..., t]
            l_hat, s_hat, diagnostic = self._separate_with_info(m_t)
            low_rank[..., t] = l_hat
            sparse[..., t] = s_hat

            if t < self.config.train_length:
                continue

            update_buffer.append(l_hat)
            diagnostic_buffer.append(diagnostic)
            if len(update_buffer) < self.config.alpha:
                continue

            update_start = t - len(update_buffer) + 1
            reported_time = self._reported_change_time(update_start, t)
            changed, update_events, score_components = self._update_subspaces(
                update_buffer,
                diagnostic_buffer,
                update_time=t,
                update_start=update_start,
                reported_time=reported_time,
            )
            self._apply_tied_modes()
            update_times.append(t)
            change_score_times.append(reported_time)
            change_score_rows.append(score_components)
            update_buffer = []
            diagnostic_buffer = []
            change_history.extend(update_events)

            ranks = tuple(base.shape[1] for base in self.bases_)
            if self.config.store_rank_history:
                rank_history.append((t, ranks))
            if changed:
                raw_change_points.append(reported_time)

        filtered_change_points, filtered_indices = self._filter_change_points(
            raw_change_points,
            change_score_times,
            change_score_rows,
        )
        change_points = filtered_change_points
        change_point_modes = self._change_point_modes(change_history, change_points)
        change_point_ms = self._frames_to_ms(change_points, n_times)
        change_scores = self._change_scores_array(change_score_rows)
        change_point_score_components = np.asarray(
            [change_score_rows[index] for index in filtered_indices],
            dtype=object,
        )

        return HoRLSLResult(
            low_rank=low_rank,
            sparse=sparse,
            change_points=change_points,
            final_bases=[basis.copy() for basis in self.bases_],
            rank_history=rank_history,
            update_times=update_times,
            change_history=change_history,
            raw_change_points=raw_change_points,
            filtered_change_points=filtered_change_points,
            change_point_ms=change_point_ms,
            change_point_modes=change_point_modes,
            change_scores=change_scores,
            change_score_times=np.asarray(change_score_times, dtype=int),
            change_point_score_components=change_point_score_components,
            solver_params=self._solver_params(),
            input_layout=PAPER_TIME_LAST_LAYOUT,
            output_layout=PAPER_TIME_LAST_LAYOUT,
            internal_layout=PAPER_TIME_LAST_LAYOUT,
            original_shape=tuple(sequence.shape),
            internal_shape=tuple(sequence.shape),
            initial_ranks=tuple(int(rank) for rank in (self.initial_ranks_ or ())),
            initial_thresholds=tuple(float(value) for value in self.thresholds_),
            config=asdict(self.config),
        )

    def fit_transform_subject_first(self, tensor: np.ndarray) -> HoRLSLResult:
        """Run Ho-RLSL on `(subjects, nodes, nodes, time)` data.

        The internal sequence is reordered to `(nodes, nodes, subjects, time)`
        to match the paper's `N x N x subjects` tensor at each time point. The
        returned low-rank and sparse tensors are converted back to the original
        subject-first shape.
        """
        tensor = validate_subject_first_tensor(tensor)
        sequence = subject_first_to_paper_time_last(tensor)
        result = self.fit_transform(sequence)
        result.low_rank = np.transpose(result.low_rank, (2, 0, 1, 3))
        result.sparse = np.transpose(result.sparse, (2, 0, 1, 3))
        result.input_layout = SUBJECT_FIRST_LAYOUT
        result.output_layout = SUBJECT_FIRST_LAYOUT
        result.original_shape = tuple(tensor.shape)
        result.internal_shape = tuple(sequence.shape)
        return result

    def _validate_config(self, n_modes: int) -> None:
        if self.config.train_length < 1:
            raise ValueError("train_length must be positive")
        if self.config.alpha < 1:
            raise ValueError("alpha must be positive")
        if self.config.change_point_position not in {"start", "midpoint", "end"}:
            raise ValueError("change_point_position must be 'start', 'midpoint', or 'end'")
        if self.config.change_point_modes is not None:
            for mode in self.config.change_point_modes:
                if not (0 <= int(mode) < n_modes):
                    raise ValueError(f"change_point_modes entries must be in [0, {n_modes - 1}]")
        if self.config.init_method not in {"hosvd"}:
            raise ValueError("init_method must be 'hosvd'")
        if self.config.threshold_mode not in {"relative", "absolute"}:
            raise ValueError("threshold_mode must be 'relative' or 'absolute'")
        if self.config.sparse_solver not in {"fista", "fista_l1", "proxy", "gtcs_s_omp"}:
            raise ValueError("sparse_solver must be 'fista', 'fista_l1', 'proxy', or 'gtcs_s_omp'")
        if self.config.lambda_sparse is not None and self.config.lambda_sparse < 0:
            raise ValueError("lambda_sparse must be non-negative")
        if self.config.epsilon is not None and self.config.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.config.epsilon_mode not in {"absolute", "relative"}:
            raise ValueError("epsilon_mode must be 'absolute' or 'relative'")
        if self.config.fista_max_iter < 1:
            raise ValueError("fista_max_iter must be positive")
        if self.config.fista_tol < 0:
            raise ValueError("fista_tol must be non-negative")
        if self.config.fista_step_size <= 0:
            raise ValueError("fista_step_size must be positive")
        if self.config.sparsity < 1:
            raise ValueError("sparsity must be positive")
        if self.config.max_atoms is not None and self.config.max_atoms < 1:
            raise ValueError("max_atoms must be positive")
        if self.config.residual_tol < 0:
            raise ValueError("residual_tol must be non-negative")
        if self.config.sparse_threshold_mode not in {"mad", "relative", "absolute"}:
            raise ValueError("sparse_threshold_mode must be 'mad', 'relative', or 'absolute'")
        if self.config.threshold_method not in {"mad", "percentile"}:
            raise ValueError("threshold_method must be 'mad' or 'percentile'")
        if self.config.threshold_k < 0:
            raise ValueError("threshold_k must be non-negative")
        if self.config.min_cp_distance_ms < 0:
            raise ValueError("min_cp_distance_ms must be non-negative")
        if self.config.score_smoothing_ms < 0:
            raise ValueError("score_smoothing_ms must be non-negative")
        if self.config.sampling_rate is not None and self.config.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive when provided")
        if self.config.target_window_ms is not None:
            if len(self.config.target_window_ms) != 2:
                raise ValueError("target_window_ms must contain start and end in milliseconds")
            if self.config.target_window_ms[0] > self.config.target_window_ms[1]:
                raise ValueError("target_window_ms start must be <= end")
        if len(self.config.mode_weights) < n_modes:
            raise ValueError(f"mode_weights must contain at least {n_modes} entries")
        if self.config.max_ranks is not None and len(self.config.max_ranks) != n_modes:
            raise ValueError(f"max_ranks must contain {n_modes} entries")
        if self.config.tie_symmetric_modes is not None:
            source, target = self.config.tie_symmetric_modes
            if not (0 <= source < n_modes and 0 <= target < n_modes):
                raise ValueError(f"tie_symmetric_modes entries must be in [0, {n_modes - 1}]")

    def _reported_change_time(self, update_start: int, update_end: int) -> int:
        if self.config.change_point_position == "start":
            return int(update_start)
        if self.config.change_point_position == "end":
            return int(update_end)
        return int(round((int(update_start) + int(update_end)) / 2))

    def _mode_counts_for_change_point(self, mode: int) -> bool:
        modes = self.config.change_point_modes
        return modes is None or int(mode) in {int(item) for item in modes}

    def _initial_subspaces(self, train: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        if self.config.init_method == "hosvd":
            return self._initial_subspaces_hosvd(train)
        raise ValueError(f"Unsupported init_method: {self.config.init_method}")

    def _initial_subspaces_hosvd(self, train: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Initialize mode subspaces from the training tensor via HOSVD.

        The paper writes the training block as
        `M_train = C x_1 P0_1 x_2 P0_2 x_3 P0_3 x_4 P0_4`.
        HOSVD obtains each `P0_i` from the left singular vectors of the
        corresponding mode unfolding. Only the first three factors are used as
        online low-rank subspaces; the fourth factor is the training-time mode.
        """
        train = np.asarray(train, dtype=float)
        full_factors = []
        singular_values_by_mode = []

        for mode in range(train.ndim):
            data = unfold(train, mode)
            if self.config.normalize_unfoldings:
                data = data / np.sqrt(max(data.shape[1], 1))
            u, singular_values, _ = np.linalg.svd(data, full_matrices=False)
            full_factors.append(u)
            singular_values_by_mode.append(singular_values)

        self.initial_full_factors_ = full_factors
        self.initial_time_basis_ = full_factors[-1]
        self.initial_singular_values_ = singular_values_by_mode
        self.initial_core_ = multi_mode_dot(
            train,
            [factor.T for factor in full_factors],
            modes=range(train.ndim),
        )

        bases = []
        thresholds = []
        for mode in range(train.ndim - 1):
            u = full_factors[mode]
            singular_values = singular_values_by_mode[mode]
            threshold = self._threshold_from_values(singular_values)
            max_rank = self._max_rank(mode)
            keep = singular_values >= threshold
            if max_rank is not None:
                keep_indices = np.flatnonzero(keep)[:max_rank]
            else:
                keep_indices = np.flatnonzero(keep)
            if keep_indices.size == 0 and singular_values.size > 0:
                keep_indices = np.array([0])
            bases.append(orthonormalize(u[:, keep_indices]))
            thresholds.append(float(threshold))
        return bases, thresholds

    def _threshold_from_values(self, values: np.ndarray) -> float:
        if values.size == 0:
            return float(self.config.sigma_min)
        if self.config.threshold_mode == "relative":
            return float(self.config.sigma_min * values[0])
        return float(self.config.sigma_min)

    def _sparse_threshold(self, residual: np.ndarray) -> float:
        mode = self.config.sparse_threshold_mode
        value = float(self.config.sparse_threshold)
        abs_residual = np.abs(residual)

        if mode == "absolute":
            return value
        if mode == "relative":
            return value * float(abs_residual.max(initial=0.0))

        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= np.finfo(float).eps:
            robust_sigma = float(abs_residual.std())
        return value * robust_sigma

    def _separate(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low_hat, sparse_hat, _ = self._separate_with_info(observed)
        return low_hat, sparse_hat

    def _separate_with_info(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        projectors = self._orthogonal_projectors()
        projected_measurement = self._project_orthogonal(observed, projectors)
        sparse_hat, solver_info = self._recover_sparse_with_info(projected_measurement, projectors)
        low_hat = observed - sparse_hat
        low_hat = self._sanitize_low_rank(low_hat)
        sparse_hat = observed - low_hat
        support = np.flatnonzero(np.abs(sparse_hat).ravel() > np.finfo(float).eps)
        projected_norm = float(np.linalg.norm(projected_measurement))
        reconstruction_error = float(solver_info.get("residual_norm", 0.0)) / max(projected_norm, 1.0)
        info = {
            **solver_info,
            "support": support.astype(int),
            "support_size": int(support.size),
            "sparse_energy": float(np.linalg.norm(sparse_hat)),
            "low_rank_energy": float(np.linalg.norm(low_hat)),
            "projected_norm": projected_norm,
            "reconstruction_error": reconstruction_error,
        }
        return low_hat, sparse_hat, info

    def _orthogonal_projectors(self) -> list[np.ndarray]:
        """Return phi_i = I - P_i P_i^T for each tracked mode."""
        if self.bases_ is None:
            raise RuntimeError("Subspaces have not been initialized")

        projectors = []
        for basis in self.bases_:
            identity = np.eye(basis.shape[0], dtype=float)
            if basis.shape[1] == 0:
                projectors.append(identity)
            else:
                projectors.append(identity - basis @ basis.T)
        return projectors

    def _project_orthogonal(self, tensor: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        """Compute Y_t = M_t x_1 phi_1 x_2 phi_2 x_3 phi_3."""
        return multi_mode_dot(tensor, projectors, modes=range(len(projectors)))

    def _backproject_sparse_proxy(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> np.ndarray:
        """Apply A* to a projected tensor, where A(S)=S x_i phi_i."""
        adjoint_projectors = [projector.T for projector in projectors]
        return multi_mode_dot(
            projected_measurement,
            adjoint_projectors,
            modes=range(len(adjoint_projectors)),
        )

    def _recover_sparse(self, projected_measurement: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        sparse, _ = self._recover_sparse_with_info(projected_measurement, projectors)
        return sparse

    def _recover_sparse_with_info(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, object]]:
        if self.config.sparse_solver == "proxy":
            sparse_proxy = self._backproject_sparse_proxy(projected_measurement, projectors)
            sparse = soft_threshold(sparse_proxy, self._sparse_threshold(sparse_proxy))
            residual = self._project_orthogonal(sparse, projectors) - projected_measurement
            return sparse, {
                "solver": "proxy",
                "residual_norm": float(np.linalg.norm(residual)),
                "selected_atoms": np.flatnonzero(np.abs(sparse).ravel() > np.finfo(float).eps).astype(int),
            }
        if self.config.sparse_solver == "gtcs_s_omp":
            return self._recover_sparse_gtcs_s_omp(projected_measurement, projectors)
        sparse = self._recover_sparse_fista(projected_measurement, projectors)
        residual = self._project_orthogonal(sparse, projectors) - projected_measurement
        return sparse, {
            "solver": "fista_l1",
            "residual_norm": float(np.linalg.norm(residual)),
            "selected_atoms": np.flatnonzero(np.abs(sparse).ravel() > np.finfo(float).eps).astype(int),
        }

    def _lambda_sparse(self, projected_measurement: np.ndarray, projectors: list[np.ndarray]) -> float:
        if self.config.lambda_sparse is not None:
            return float(self.config.lambda_sparse)
        sparse_proxy = self._backproject_sparse_proxy(projected_measurement, projectors)
        return self._sparse_threshold(sparse_proxy)

    def _epsilon_value(self, projected_measurement: np.ndarray) -> float | None:
        if self.config.epsilon is None:
            return None
        epsilon = float(self.config.epsilon)
        if self.config.epsilon_mode == "relative":
            return epsilon * float(np.linalg.norm(projected_measurement))
        return epsilon

    def _recover_sparse_fista(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> np.ndarray:
        """L1 sparse recovery proxy for GTCS-S using FISTA.

        Solves `0.5 * ||A(S) - Y||_F^2 + lambda * ||S||_1`, where
        `A(S) = S x_1 phi_1 x_2 phi_2 x_3 phi_3`.
        """
        sparse = np.zeros_like(projected_measurement, dtype=float)
        momentum = sparse.copy()
        t_value = 1.0
        step = float(self.config.fista_step_size)
        lambda_step = step * self._lambda_sparse(projected_measurement, projectors)
        epsilon = self._epsilon_value(projected_measurement)

        for _ in range(self.config.fista_max_iter):
            residual = self._project_orthogonal(momentum, projectors) - projected_measurement
            gradient = self._backproject_sparse_proxy(residual, projectors)
            next_sparse = soft_threshold(momentum - step * gradient, lambda_step)

            denominator = max(float(np.linalg.norm(sparse)), 1.0)
            relative_change = float(np.linalg.norm(next_sparse - sparse) / denominator)

            next_t_value = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))
            momentum = next_sparse + ((t_value - 1.0) / next_t_value) * (next_sparse - sparse)
            sparse = next_sparse
            t_value = next_t_value

            if epsilon is not None:
                constrained_residual = self._project_orthogonal(sparse, projectors) - projected_measurement
                if float(np.linalg.norm(constrained_residual)) <= epsilon:
                    break

            if relative_change <= self.config.fista_tol:
                break

        return sparse

    def _recover_sparse_gtcs_s_omp(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, object]]:
        """GTCS-S-inspired greedy sparse recovery.

        This is not the MATLAB GTCS-S routine from the paper. It keeps the
        same compressed tensor measurement model, then performs an OMP-style
        sparse pursuit over canonical tensor atoms under `A(S)=S x_i phi_i`.
        """
        residual = np.asarray(projected_measurement, dtype=float).copy()
        target = projected_measurement.ravel()
        shape = projected_measurement.shape
        max_atoms = self.config.max_atoms or self.config.sparsity
        max_atoms = int(min(max_atoms, self.config.sparsity, projected_measurement.size))
        selected: list[int] = []
        residual_history = [float(np.linalg.norm(residual))]
        coefficients = np.asarray([], dtype=float)

        for _ in range(max_atoms):
            proxy = self._backproject_sparse_proxy(residual, projectors).ravel()
            if selected:
                proxy[np.asarray(selected, dtype=int)] = 0.0
            atom_index = int(np.argmax(np.abs(proxy)))
            if abs(float(proxy[atom_index])) <= np.finfo(float).eps:
                break

            selected.append(atom_index)
            dictionary = np.column_stack(
                [self._projected_canonical_atom(index, shape, projectors).ravel() for index in selected]
            )
            coefficients, *_ = np.linalg.lstsq(dictionary, target, rcond=None)
            residual_vector = target - dictionary @ coefficients
            residual = residual_vector.reshape(shape)
            residual_norm = float(np.linalg.norm(residual))
            residual_history.append(residual_norm)
            target_norm = max(float(np.linalg.norm(target)), 1.0)
            if residual_norm / target_norm <= self.config.residual_tol:
                break

        sparse = np.zeros(shape, dtype=float)
        if selected:
            sparse.ravel()[np.asarray(selected, dtype=int)] = coefficients

        return sparse, {
            "solver": "gtcs_s_omp",
            "residual_norm": float(np.linalg.norm(residual)),
            "selected_atoms": np.asarray(selected, dtype=int),
            "residual_history": np.asarray(residual_history, dtype=float),
            "coefficients": np.asarray(coefficients, dtype=float),
        }

    def _projected_canonical_atom(
        self,
        flat_index: int,
        shape: tuple[int, ...],
        projectors: list[np.ndarray],
    ) -> np.ndarray:
        coords = np.unravel_index(int(flat_index), shape)
        atom = projectors[0][:, coords[0]]
        result = atom
        for mode, projector in enumerate(projectors[1:], start=1):
            result = np.multiply.outer(result, projector[:, coords[mode]])
        return np.asarray(result, dtype=float)

    def _sanitize_low_rank(self, low_rank: np.ndarray) -> np.ndarray:
        low_rank = np.asarray(low_rank, dtype=float).copy()
        low_rank[~np.isfinite(low_rank)] = 0.0

        if self.config.enforce_symmetric_connectivity and low_rank.ndim >= 2 and low_rank.shape[0] == low_rank.shape[1]:
            low_rank = 0.5 * (low_rank + np.swapaxes(low_rank, 0, 1))

        if self.config.zero_diagonal and low_rank.ndim >= 2 and low_rank.shape[0] == low_rank.shape[1]:
            diagonal = np.arange(low_rank.shape[0])
            low_rank[diagonal, diagonal, ...] = 0.0

        return low_rank

    def _project_low_rank(self, tensor: np.ndarray) -> np.ndarray:
        if self.bases_ is None:
            raise RuntimeError("Subspaces have not been initialized")
        core = tensor
        for mode, basis in enumerate(self.bases_):
            core = mode_dot(core, basis.T, mode)
        result = core
        for mode, basis in enumerate(self.bases_):
            result = mode_dot(result, basis, mode)
        return result

    def _score_update(
        self,
        diagnostics: list[dict[str, object]],
        tensors: list[np.ndarray],
        previous_bases: list[np.ndarray],
        current_bases: list[np.ndarray],
        update_events: list[dict[str, int]],
    ) -> dict[str, float]:
        reconstruction_error_score = float(
            np.mean([float(item.get("reconstruction_error", 0.0)) for item in diagnostics])
        ) if diagnostics else 0.0
        support_change_score = self._support_change_score(diagnostics)
        subspace_angle_score = self._weighted_subspace_distance(previous_bases, current_bases)
        mode_energy_score = self._mode_energy_score(tensors, current_bases)
        direction_update_score = self._direction_update_score(update_events)
        final_score = float(
            0.30 * reconstruction_error_score
            + 0.20 * support_change_score
            + 0.30 * subspace_angle_score
            + 0.15 * mode_energy_score
            + 0.05 * direction_update_score
        )
        dominant_mode = self._dominant_changed_mode(update_events, previous_bases, current_bases)
        return {
            "reconstruction_error_score": reconstruction_error_score,
            "support_change_score": support_change_score,
            "subspace_angle_score": subspace_angle_score,
            "mode_energy_score": mode_energy_score,
            "direction_update_score": direction_update_score,
            "final_score": final_score,
            "adaptive_threshold": 0.0,
            "dominant_mode": float(dominant_mode),
        }

    def _support_change_score(self, diagnostics: list[dict[str, object]]) -> float:
        supports = [set(np.asarray(item.get("support", []), dtype=int).ravel().tolist()) for item in diagnostics]
        if len(supports) < 2:
            return 0.0
        scores = []
        previous = supports[0]
        for current in supports[1:]:
            union = previous | current
            if not union:
                scores.append(0.0)
            else:
                scores.append(1.0 - len(previous & current) / len(union))
            previous = current
        return float(np.mean(scores)) if scores else 0.0

    def _mode_weight(self, mode: int) -> float:
        if mode < len(self.config.mode_weights):
            return float(self.config.mode_weights[mode])
        return 1.0

    def _weighted_subspace_distance(
        self,
        previous_bases: list[np.ndarray],
        current_bases: list[np.ndarray],
    ) -> float:
        weighted = []
        weights = []
        for mode, (old_basis, new_basis) in enumerate(zip(previous_bases, current_bases)):
            weight = self._mode_weight(mode)
            weighted.append(weight * subspace_distance(old_basis, new_basis))
            weights.append(weight)
        return float(np.sum(weighted) / max(np.sum(weights), np.finfo(float).eps)) if weighted else 0.0

    def _mode_energy_score(self, tensors: list[np.ndarray], bases: list[np.ndarray]) -> float:
        if not tensors:
            return 0.0
        mode_scores = []
        weights = []
        for mode, basis in enumerate(bases):
            residual_ratios = []
            for tensor in tensors:
                data = unfold(tensor, mode)
                if basis.shape[1] > 0:
                    residual = data - basis @ (basis.T @ data)
                else:
                    residual = data
                residual_ratios.append(float(np.linalg.norm(residual) / max(np.linalg.norm(data), 1.0)))
            weight = self._mode_weight(mode)
            mode_scores.append(weight * float(np.mean(residual_ratios)))
            weights.append(weight)
        return float(np.sum(mode_scores) / max(np.sum(weights), np.finfo(float).eps)) if mode_scores else 0.0

    def _direction_update_score(self, update_events: list[dict[str, int]]) -> float:
        if not update_events:
            return 0.0
        score = 0.0
        total_weight = 0.0
        for event in update_events:
            mode = int(event.get("mode", 0))
            weight = self._mode_weight(mode)
            score += weight * float(int(event.get("deleted", 0)) + int(event.get("added", 0)))
            total_weight += weight
        return float(score / max(total_weight, 1.0))

    def _dominant_changed_mode(
        self,
        update_events: list[dict[str, int]],
        previous_bases: list[np.ndarray],
        current_bases: list[np.ndarray],
    ) -> int:
        candidates = []
        for event in update_events:
            if int(event.get("changed", 0)) == 0:
                continue
            mode = int(event.get("mode", -1))
            if mode < 0 or mode >= len(current_bases):
                continue
            score = (
                self._mode_weight(mode)
                * (1.0 + int(event.get("deleted", 0)) + int(event.get("added", 0)))
                * max(subspace_distance(previous_bases[mode], current_bases[mode]), np.finfo(float).eps)
            )
            candidates.append((score, mode))
        if not candidates:
            return -1
        return int(max(candidates, key=lambda item: item[0])[1])

    def _change_scores_array(self, rows: list[dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 7), dtype=float)
        keys = (
            "final_score",
            "reconstruction_error_score",
            "support_change_score",
            "subspace_angle_score",
            "mode_energy_score",
            "direction_update_score",
            "dominant_mode",
        )
        return np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=float)

    def _filter_change_points(
        self,
        raw_change_points: list[int],
        score_times: list[int],
        score_rows: list[dict[str, float]],
    ) -> tuple[list[int], list[int]]:
        if not raw_change_points or not score_times or not score_rows:
            return [], []

        times = np.asarray(score_times, dtype=int)
        scores = np.asarray([float(row.get("final_score", 0.0)) for row in score_rows], dtype=float)
        smoothed = self._smooth_scores(scores)
        threshold = self._adaptive_score_threshold(smoothed)
        for row, smooth_score in zip(score_rows, smoothed):
            row["smoothed_final_score"] = float(smooth_score)
            row["adaptive_threshold"] = float(threshold)

        raw_set = {int(point) for point in raw_change_points}
        candidate_indices = [
            index for index, point in enumerate(times)
            if int(point) in raw_set and smoothed[index] >= threshold and self._is_local_peak(smoothed, index)
        ]
        if not candidate_indices:
            candidate_indices = [
                index for index, point in enumerate(times)
                if int(point) in raw_set and smoothed[index] >= threshold
            ]
        if not candidate_indices and raw_set:
            raw_indices = [index for index, point in enumerate(times) if int(point) in raw_set]
            if raw_indices:
                candidate_indices = [max(raw_indices, key=lambda index: smoothed[index])]

        target_context = self._target_context_indices(times, raw_set, smoothed)
        if target_context:
            candidate_indices = target_context

        min_distance = self._min_cp_distance_frames()
        selected: list[int] = []
        for index in sorted(candidate_indices, key=lambda item: smoothed[item], reverse=True):
            point = int(times[index])
            if all(abs(point - int(times[kept])) >= min_distance for kept in selected):
                selected.append(index)
        selected.sort(key=lambda index: int(times[index]))
        return [int(times[index]) for index in selected], selected

    def _target_context_indices(
        self,
        times: np.ndarray,
        raw_set: set[int],
        scores: np.ndarray,
    ) -> list[int]:
        if self.config.target_window_ms is None or not raw_set:
            return []
        start_ms, end_ms = self.config.target_window_ms
        raw_indices = [index for index, point in enumerate(times) if int(point) in raw_set]
        if not raw_indices:
            return []
        target_indices = [
            index for index in raw_indices
            if float(start_ms) <= self._frame_to_ms(int(times[index])) <= float(end_ms)
        ]
        if not target_indices:
            return []

        anchor = min(
            target_indices,
            key=lambda index: (
                abs(self._frame_to_ms(int(times[index])) - float(self.config.target_anchor_ms)),
                -float(scores[index]),
            ),
        )
        before = [index for index in raw_indices if int(times[index]) < int(times[anchor])]
        after = [index for index in raw_indices if int(times[index]) > int(times[anchor])]
        context = []
        if before:
            context.append(max(before, key=lambda index: int(times[index])))
        context.append(anchor)
        if after:
            context.append(min(after, key=lambda index: int(times[index])))
        return sorted(set(context), key=lambda index: int(times[index]))

    def _smooth_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).ravel()
        if scores.size == 0:
            return scores
        width = self._score_smoothing_windows()
        if width <= 1 or scores.size < width:
            return scores.copy()
        if width % 2 == 0:
            width += 1
        kernel = np.ones(width, dtype=float) / float(width)
        padded = np.pad(scores, width // 2, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _score_smoothing_windows(self) -> int:
        if self.config.sampling_rate is None or self.config.score_smoothing_ms <= 0:
            return 1
        frames = (float(self.config.score_smoothing_ms) / 1000.0) * float(self.config.sampling_rate)
        return max(1, int(round(frames / max(int(self.config.alpha), 1))))

    def _adaptive_score_threshold(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float).ravel()
        if scores.size == 0:
            return np.inf
        if self.config.threshold_method == "percentile":
            percentile = float(np.clip(self.config.threshold_k, 0.0, 100.0))
            return float(np.percentile(scores, percentile))
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= np.finfo(float).eps:
            robust_sigma = float(np.std(scores))
        return float(median + float(self.config.threshold_k) * robust_sigma)

    def _is_local_peak(self, scores: np.ndarray, index: int) -> bool:
        left = scores[index - 1] if index > 0 else -np.inf
        right = scores[index + 1] if index + 1 < scores.size else -np.inf
        return bool(scores[index] >= left and scores[index] >= right)

    def _min_cp_distance_frames(self) -> int:
        if self.config.sampling_rate is None:
            return max(1, int(self.config.alpha))
        return max(1, int(round(float(self.config.min_cp_distance_ms) * float(self.config.sampling_rate) / 1000.0)))

    def _frames_to_ms(self, points: list[int], n_times: int) -> np.ndarray:
        if not points:
            return np.asarray([], dtype=float)
        if self.config.sampling_rate is not None and self.config.sampling_rate > 0:
            center = (int(n_times) - 1) / 2.0
            return (np.asarray(points, dtype=float) - center) * 1000.0 / float(self.config.sampling_rate)
        return np.linspace(-1000.0, 1000.0, int(n_times))[np.asarray(points, dtype=int)]

    def _frame_to_ms(self, point: int) -> float:
        if self.config.sampling_rate is not None and self.config.sampling_rate > 0:
            return float(-1000.0 + int(point) * 1000.0 / float(self.config.sampling_rate))
        return float(point)

    def _change_point_modes(self, change_history: list[dict[str, int]], change_points: list[int]) -> list[int]:
        modes: list[int] = []
        for point in change_points:
            events = [
                event for event in change_history
                if int(event.get("reported_time", -1)) == int(point) and int(event.get("changed", 0))
            ]
            if not events:
                modes.append(-1)
                continue
            best = max(
                events,
                key=lambda event: self._mode_weight(int(event.get("mode", 0)))
                * (1 + int(event.get("deleted", 0)) + int(event.get("added", 0))),
            )
            modes.append(int(best.get("mode", -1)))
        return modes

    def _solver_params(self) -> dict[str, object]:
        return {
            "sparse_solver": self.config.sparse_solver,
            "sparsity": int(self.config.sparsity),
            "residual_tol": float(self.config.residual_tol),
            "max_atoms": None if self.config.max_atoms is None else int(self.config.max_atoms),
            "fista_max_iter": int(self.config.fista_max_iter),
            "fista_tol": float(self.config.fista_tol),
            "fista_step_size": float(self.config.fista_step_size),
            "lambda_sparse": self.config.lambda_sparse,
            "epsilon": self.config.epsilon,
            "epsilon_mode": self.config.epsilon_mode,
        }

    def _update_subspaces(
        self,
        tensors: list[np.ndarray],
        diagnostics: list[dict[str, object]],
        update_time: int,
        update_start: int,
        reported_time: int,
    ) -> tuple[bool, list[dict[str, int]], dict[str, float]]:
        if self.bases_ is None or self.thresholds_ is None:
            raise RuntimeError("Subspaces have not been initialized")

        previous_bases = [basis.copy() for basis in self.bases_]
        changed_by_direction_update = False
        update_events: list[dict[str, int]] = []
        for mode in range(len(self.bases_)):
            if self._is_tied_target(mode):
                continue

            data = np.concatenate([unfold(tensor, mode) for tensor in tensors], axis=1)
            basis = self.bases_[mode]
            threshold = self.thresholds_[mode]
            old_rank = basis.shape[1]

            basis, deleted_count = self._delete_directions(data, basis, threshold)
            basis, added_count = self._add_directions(data, basis, threshold, mode)

            self.bases_[mode] = basis
            changed = deleted_count > 0 or added_count > 0
            counts_for_change_point = self._mode_counts_for_change_point(mode)
            changed_by_direction_update = changed_by_direction_update or (changed and counts_for_change_point)
            update_events.append(
                {
                    "time": int(update_time),
                    "window_start": int(update_start),
                    "reported_time": int(reported_time),
                    "mode": int(mode),
                    "old_rank": int(old_rank),
                    "new_rank": int(basis.shape[1]),
                    "deleted": int(deleted_count),
                    "added": int(added_count),
                    "changed": int(changed),
                    "counts_for_change_point": int(counts_for_change_point),
                }
            )

        self._apply_tied_modes()
        tied = self.config.tie_symmetric_modes
        if tied is not None:
            source, target = tied
            if source < len(self.bases_) and target < len(self.bases_):
                target_changed = subspace_distance(previous_bases[target], self.bases_[target]) > 0.0
                source_changed = any(event["mode"] == source and event["changed"] for event in update_events)
                if target_changed and source_changed:
                    update_events.append(
                        {
                            "time": int(update_time),
                            "window_start": int(update_start),
                            "reported_time": int(reported_time),
                            "mode": int(target),
                            "old_rank": int(previous_bases[target].shape[1]),
                            "new_rank": int(self.bases_[target].shape[1]),
                            "deleted": 0,
                            "added": 0,
                            "changed": 1,
                            "counts_for_change_point": int(self._mode_counts_for_change_point(target)),
                            "tied_to": int(source),
                        }
                    )
        score_components = self._score_update(
            diagnostics,
            tensors,
            previous_bases,
            self.bases_,
            update_events,
        )
        final_score = float(score_components["final_score"])
        threshold = float(score_components["adaptive_threshold"])
        changed_by_score = changed_by_direction_update and final_score >= threshold

        for event in update_events:
            event["final_score_scaled"] = int(round(final_score * 1_000_000))
            event["adaptive_threshold_scaled"] = int(round(threshold * 1_000_000))

        if self.config.use_paper_change_rule:
            return changed_by_score, update_events, score_components

        distance = max(
            subspace_distance(old_basis, new_basis)
            for old_basis, new_basis in zip(previous_bases, self.bases_)
        )
        changed_by_distance = distance >= self.config.subspace_change_threshold
        if update_events:
            update_events[-1]["subspace_distance_scaled"] = int(round(distance * 1_000_000))
        return changed_by_distance and final_score >= threshold, update_events, score_components

    def _delete_directions(
        self,
        data: np.ndarray,
        basis: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, int]:
        if basis.shape[1] == 0:
            return basis, 0

        coefficients = basis.T @ data
        eigenvalues = np.mean(coefficients * coefficients, axis=1)
        keep = eigenvalues >= threshold * threshold
        if not np.any(keep):
            keep[np.argmax(eigenvalues)] = True

        deleted_count = int(np.count_nonzero(~keep))
        return orthonormalize(basis[:, keep]), deleted_count

    def _add_directions(
        self,
        data: np.ndarray,
        basis: np.ndarray,
        threshold: float,
        mode: int,
    ) -> tuple[np.ndarray, int]:
        if basis.shape[1] > 0:
            projected = data - basis @ (basis.T @ data)
        else:
            projected = data

        projected = projected / np.sqrt(max(projected.shape[1], 1))
        u, singular_values, _ = np.linalg.svd(projected, full_matrices=False)
        eigenvalues = singular_values * singular_values
        add_indices = np.flatnonzero(eigenvalues >= threshold * threshold)
        if add_indices.size == 0:
            return basis, 0

        max_rank = self._max_rank(mode)
        if max_rank is not None:
            remaining = max(max_rank - basis.shape[1], 0)
            add_indices = add_indices[:remaining]
        if add_indices.size == 0:
            return basis, 0

        updated = np.column_stack([basis, u[:, add_indices]])
        return orthonormalize(updated), int(add_indices.size)

    def _max_rank(self, mode: int) -> int | None:
        if self.config.max_ranks is None:
            return None
        return self.config.max_ranks[mode]

    def _is_tied_target(self, mode: int) -> bool:
        tied = self.config.tie_symmetric_modes
        return tied is not None and mode == tied[1]

    def _apply_tied_modes(self) -> None:
        tied = self.config.tie_symmetric_modes
        if tied is None or self.bases_ is None:
            return
        source, target = tied
        if source >= len(self.bases_) or target >= len(self.bases_):
            return
        if self.bases_[source].shape[0] != self.bases_[target].shape[0]:
            return
        self.bases_[target] = self.bases_[source].copy()


def fit_transform_subject_first(
    tensor: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    return HoRLSL(config).fit_transform_subject_first(tensor)


def fit_transform_connectivity(
    tensor: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    """Alias for repo-native `(subjects, nodes, nodes, time)` connectivity tensors."""
    return fit_transform_subject_first(tensor, config)


def fit_transform_time_last(
    sequence: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    return HoRLSL(config).fit_transform(sequence)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Ho-RLSL on a time-last connectivity tensor.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("outputs/tensor_4d_1024/tensor_incorrect_4d.npy"))
    parser.add_argument("--output", type=Path, default=Path("ho_rlsl_results.npz"))
    parser.add_argument(
        "--layout",
        choices=["subject-first", "paper"],
        default="subject-first",
        help=(
            "subject-first means (subjects, nodes, nodes, time); "
            "paper means (nodes, nodes, subjects, time)"
        ),
    )
    parser.add_argument("--train-length", type=int, default=80)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--sigma-min", type=float, default=0.11)
    parser.add_argument("--init-method", choices=["hosvd"], default="hosvd")
    parser.add_argument("--threshold-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument(
        "--raw-hosvd-singular-values",
        action="store_true",
        help="Use raw HOSVD singular values instead of covariance-scaled values for thresholding.",
    )
    parser.add_argument("--sparse-threshold", type=float, default=3.0)
    parser.add_argument("--sparse-threshold-mode", choices=["mad", "relative", "absolute"], default="mad")
    parser.add_argument(
        "--sparse-solver",
        choices=["gtcs_s_omp"],
        default="gtcs_s_omp",
    )
    parser.add_argument("--lambda-sparse", type=float, default=None)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional residual constraint for sparse recovery: ||A(S)-Y||_F <= epsilon.",
    )
    parser.add_argument("--epsilon-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--fista-max-iter", type=int, default=20)
    parser.add_argument("--fista-tol", type=float, default=1e-4)
    parser.add_argument("--fista-step-size", type=float, default=1.0)
    parser.add_argument("--sparsity", type=int, default=8)
    parser.add_argument("--residual-tol", type=float, default=1e-3)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument(
        "--max-ranks",
        default="15,15,10",
        help="Optional comma-separated ranks for modes 0,1,2, e.g. '10,10,10' or '10,10,None'.",
    )
    parser.add_argument(
        "--tie-symmetric-modes",
        default="0,1",
        help="Mode pair to tie for symmetric connectivity, e.g. '0,1', or 'None' to disable.",
    )
    parser.add_argument(
        "--no-symmetric-connectivity",
        action="store_true",
        help="Do not symmetrize low-rank node-node connectivity estimates.",
    )
    parser.add_argument(
        "--keep-diagonal",
        action="store_true",
        help="Do not force low-rank node-node connectivity diagonals to zero.",
    )
    parser.add_argument(
        "--subspace-distance-change-rule",
        action="store_true",
        help="Use the fallback subspace-distance change rule instead of the paper add/delete rule.",
    )
    parser.add_argument(
        "--change-point-position",
        choices=["start", "midpoint", "end"],
        default="midpoint",
        help=(
            "Report detected changes at the update window start, midpoint, or end. "
            "The update time is still stored separately in update_times."
        ),
    )
    parser.add_argument(
        "--change-point-modes",
        default=None,
        help=(
            "Optional comma-separated internal modes allowed to create change points. "
            "For paper layout, modes 0 and 1 are connectivity and mode 2 is subjects."
        ),
    )
    parser.add_argument("--min-cp-distance-ms", type=float, default=50.0)
    parser.add_argument("--score-smoothing-ms", type=float, default=25.0)
    parser.add_argument("--threshold-method", choices=["mad", "percentile"], default="mad")
    parser.add_argument("--threshold-k", type=float, default=3.0)
    parser.add_argument("--target-window-start-ms", type=float, default=-150.0,
                        help="Start of target window in ms (default: -150)")
    parser.add_argument("--target-window-end-ms", type=float, default=150.0,
                        help="End of target window in ms (default: 150)")
    parser.add_argument("--no-target-window", action="store_true",
                        help="Disable target window filtering.")
    parser.add_argument("--target-anchor-ms", type=float, default=50.0)
    parser.add_argument(
        "--mode-weights",
        default="1.0,1.0,0.4",
        help="Comma-separated weights for internal modes 0,1,2.",
    )
    parser.add_argument("--sampling-rate", type=float, default=1024.0)
    args = parser.parse_args()

    x = np.load(args.input)
    max_ranks = parse_optional_int_tuple(args.max_ranks)
    tie_symmetric_modes = parse_int_pair(args.tie_symmetric_modes)
    change_point_modes = parse_optional_int_tuple(args.change_point_modes)
    if change_point_modes is not None and any(mode is None for mode in change_point_modes):
        raise ValueError("--change-point-modes must contain integers only, e.g. '0' or '0,1'")
    mode_weights = tuple(float(item.strip()) for item in args.mode_weights.split(",") if item.strip())
    target_window_ms = None
    if not args.no_target_window:
        target_window_ms = (args.target_window_start_ms, args.target_window_end_ms)
    cfg = HoRLSLConfig(
        train_length=args.train_length,
        alpha=args.alpha,
        sigma_min=args.sigma_min,
        init_method=args.init_method,
        threshold_mode=args.threshold_mode,
        normalize_unfoldings=not args.raw_hosvd_singular_values,
        sparse_solver=args.sparse_solver,
        lambda_sparse=args.lambda_sparse,
        epsilon=args.epsilon,
        epsilon_mode=args.epsilon_mode,
        fista_max_iter=args.fista_max_iter,
        fista_tol=args.fista_tol,
        fista_step_size=args.fista_step_size,
        sparsity=args.sparsity,
        residual_tol=args.residual_tol,
        max_atoms=args.max_atoms,
        sparse_threshold=args.sparse_threshold,
        sparse_threshold_mode=args.sparse_threshold_mode,
        enforce_symmetric_connectivity=not args.no_symmetric_connectivity,
        zero_diagonal=not args.keep_diagonal,
        max_ranks=max_ranks,
        tie_symmetric_modes=tie_symmetric_modes,
        use_paper_change_rule=not args.subspace_distance_change_rule,
        change_point_position=args.change_point_position,
        change_point_modes=None if change_point_modes is None else tuple(int(mode) for mode in change_point_modes),
        min_cp_distance_ms=args.min_cp_distance_ms,
        score_smoothing_ms=args.score_smoothing_ms,
        threshold_method=args.threshold_method,
        threshold_k=args.threshold_k,
        mode_weights=mode_weights,
        sampling_rate=args.sampling_rate,
        target_window_ms=target_window_ms,
        target_anchor_ms=args.target_anchor_ms,
    )
    if args.layout == "subject-first":
        out = fit_transform_subject_first(x, cfg)
    else:
        out = fit_transform_time_last(x, cfg)
    np.savez_compressed(
        args.output,
        low_rank=out.low_rank.astype(np.float32),
        sparse=out.sparse.astype(np.float32),
        change_points=np.asarray(out.change_points, dtype=int),
        raw_change_points=np.asarray(out.raw_change_points, dtype=int),
        filtered_change_points=np.asarray(out.filtered_change_points, dtype=int),
        change_point_ms=np.asarray(out.change_point_ms, dtype=float),
        change_point_modes=np.asarray(out.change_point_modes, dtype=int),
        change_scores=np.asarray(out.change_scores, dtype=float),
        change_score_times=np.asarray(out.change_score_times, dtype=int),
        change_point_score_components=np.asarray(out.change_point_score_components, dtype=object),
        sparse_solver=np.asarray(out.config.get("sparse_solver", "")),
        solver_params=np.asarray(out.solver_params, dtype=object),
        update_times=np.asarray(out.update_times, dtype=int),
        rank_history=np.asarray(out.rank_history, dtype=object),
        change_history=np.asarray(out.change_history, dtype=object),
        input_layout=np.asarray(out.input_layout),
        output_layout=np.asarray(out.output_layout),
        internal_layout=np.asarray(out.internal_layout),
        original_shape=np.asarray(out.original_shape, dtype=int),
        internal_shape=np.asarray(out.internal_shape, dtype=int),
        initial_ranks=np.asarray(out.initial_ranks, dtype=int),
        initial_thresholds=np.asarray(out.initial_thresholds, dtype=float),
        config=np.asarray(out.config, dtype=object),
    )
    print(f"Saved {args.output}")
    print(f"Detected change points: {out.change_points}")
