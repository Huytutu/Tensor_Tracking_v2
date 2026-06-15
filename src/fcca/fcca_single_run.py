"""
Run the FCCA paper-style ERN clustering pipeline in Python.

This is a structure-preserving translation of cluster_ern_fcca_paper.m:
the main function stays at the top and the helper functions follow below
in the same broad order.

Usage:
    python cluster_ern_fcca_paper.py
    python cluster_ern_fcca_paper.py 3
    python cluster_ern_fcca_paper.py 2 3 4
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.io import loadmat, savemat, whosmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPS = np.finfo(float).eps

# Chinh o day neu muon chay truc tiep tu IDE:
#   []        -> tu tim k trong cfg.kRange
#   3         -> ep ca 3 khoang Pre-ERN/ERN/Post-ERN dung k=3
#   [2, 3, 4] -> lan luot k cho Pre-ERN, ERN, Post-ERN
DEFAULT_K_INPUT: list[int] | int = []

# INPUT DU LIEU - chi can sua dong nay khi muon doi file dau vao.
INPUT_DATA_FILE: str | None = "../../outputs/ho_rlsl_results.npz"

# Frame goc 0-based trong file .npz. 4 moc nay tao 3 khoang lien tiep.
INTERVAL_BOUNDARY_FRAMES_0BASED: tuple[int, int, int, int] | None = None

OUTPUT_DIR_NAME = "../../outputs/fcca_paper_visualizations"


@dataclass
class Config:
    sampleRateHz: float = 1024.0
    epochStartMs: float = -1000.0
    analysisWindowMs: tuple[float, float] = (-800.0, 800.0)
    preStartMs: float = -800.0
    kRange: tuple[int, ...] = (2, 3, 4, 5, 6)
    selectionCriterion: str = "U"
    requestedK: Any = None
    minCommunitySize: int = 2
    maxPlotEdges: int = 35
    isolatedNodeColor: tuple[float, float, float] = (0.72, 0.72, 0.72)
    maxGraphsPerInterval: float = math.inf
    randomSeed: int = 42


@dataclass
class IntervalSpec:
    key: str
    displayName: str
    indices: np.ndarray
    frameNumbers: np.ndarray
    windowMs: np.ndarray


def cluster_ern_fcca_paper(k_input: Any = None) -> dict[str, Any]:
    """
    Chạy pipeline Fiedler Consensus Clustering Approach (FCCA) trên dữ liệu ERN.
    
    Pipeline chính gồm các bước:
    1. Tải tensor lowrank từ file dữ liệu
    2. Phát hiện Change Point để chia trạng thái Pre-ERN, ERN, Post-ERN
    3. Áp dụng FCCA để tìm cấu trúc cộng đồng tối ưu cho mỗi khoảng thời gian
    4. Lưu kết quả và vẽ biểu đồ trực quan
    """

    if k_input is None:
        k_input = []

    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    if INPUT_DATA_FILE:
        data_file = Path(INPUT_DATA_FILE)
        if not data_file.is_absolute():
            data_file = base_dir / data_file
    else:
        data_file = first_existing_file(
            [
                base_dir / "ho_rlsl_results.npz",
                base_dir.parent / "ho_rlsl_results.npz",
                Path.cwd() / "ho_rlsl_results.npz",
                base_dir / "ho_rlsl_results.mat",
                base_dir.parent / "ho_rlsl_results.mat",
                Path.cwd() / "ho_rlsl_results.mat",
            ]
        )
    if data_file is None:
        raise FileNotFoundError("Cannot find ho_rlsl_results.mat or ho_rlsl_results.npz.")
    if not data_file.is_file():
        raise FileNotFoundError(f"Cannot find data file: {data_file}")

    raw_bids_dir = first_existing_dir(
        [
            base_dir / "ERN Raw Data BIDS-Compatible",
            base_dir.parent / "ERN Raw Data BIDS-Compatible",
            Path.cwd() / "ERN Raw Data BIDS-Compatible",
            Path.cwd() / "data" / "ERN_Raw_Data_BIDS-Compatible",
            base_dir.parent.parent / "data" / "ERN_Raw_Data_BIDS-Compatible",
        ]
    )

    output_dir = base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(requestedK=k_input)
    np.random.seed(cfg.randomSeed)

    # ===== BƯỚC 1: TẢI DỮ LIỆU =====
    print("=== FCCA paper-style ERN clustering ===")
    print(f"Data file: {data_file}")
    print(f"k_input: {'[]' if is_empty_k(k_input) else mat2str(k_input)}")
    # Tải tensor kết nối (shape: subjects × nodes × nodes × frames)
    # và metadata chứa Change Point information
    tensor, metadata = load_ho_rlsl_tensor(data_file)
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 4:
        raise ValueError("Expected tensor as subjects x nodes x nodes x frames.")

    n_subjects, n_nodes, n_nodes_2, n_frames = tensor.shape
    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square adjacency matrices, got {n_nodes} x {n_nodes_2}.")

    # ===== BƯỚC 2: PHÁT HIỆN CHANGE POINT VÀ CHIA KHOẢNG THỜI GIAN =====
    # Tính toán mảng thời gian (ms) từ đầu epoch
    times_ms = cfg.epochStartMs + (np.arange(n_frames) / cfg.sampleRateHz) * 1000.0
    # Lựa chọn Change Point từ metadata để xác định ranh giới ERN
    # Thường có 2 điểm: bắt đầu ERN và kết thúc ERN
    change_points = choose_change_points_from_metadata(
        metadata, n_frames, times_ms, cfg.analysisWindowMs, INTERVAL_BOUNDARY_FRAMES_0BASED
    )
    # Tạo 3 khoảng phân tích: Pre-ERN, ERN, Post-ERN
    intervals = make_interval_specs(
        change_points, n_frames, times_ms, cfg.analysisWindowMs, cfg.preStartMs
    )
    node_xy, channel_labels, layout_note = ern_head_layout(n_nodes, raw_bids_dir)

    print(f"Tensor: {n_subjects} subjects x {n_nodes} nodes x {n_nodes} nodes x {n_frames} frames")
    print(f"Analysis window: {cfg.analysisWindowMs[0]:.1f} to {cfg.analysisWindowMs[1]:.1f} ms")
    print(f"Pre-ERN starts at {cfg.preStartMs:.1f} ms")
    if INTERVAL_BOUNDARY_FRAMES_0BASED is not None:
        print(f"Manual interval boundaries, 0-based: {mat2str(INTERVAL_BOUNDARY_FRAMES_0BASED)}")
    print(f"Change points used: {mat2str(change_points)}")
    print(f"Change-point times: {mat2str(times_ms[change_points - 1], precision=4)} ms")
    print(f"{layout_note}\n")

    results: dict[str, Any] = {
        "config": config_to_dict(cfg),
        "dataFile": str(data_file),
        "metadata": metadata,
        "changePoints": change_points,
        "changePointTimesMs": times_ms[change_points - 1],
        "timesMs": times_ms,
        "channelLabels": np.asarray(channel_labels, dtype=object),
        "nodeXY": node_xy,
        "intervals": [interval_to_result(interval) for interval in intervals],
    }

    # ===== BƯỚC 3: XỬ LÝ TỪNG KHOẢNG THỜI GIAN =====
    for interval_idx, interval in enumerate(intervals, start=1):
        print(
            f"Preparing {interval.displayName}: frames "
            f"{interval.frameNumbers[0]}-{interval.frameNumbers[-1]}, "
            f"{interval.windowMs[0]:.1f} to {interval.windowMs[1]:.1f} ms"
        )

        # Trích xuất các ma trận kết nối từ khoảng thời gian này
        # Output: mảng (nodes × nodes × subjects) chứa m đồ thị từ m đối tượng
        graphs = interval_graphs_from_tensor(tensor, interval.indices, cfg)
        # Tính ma trận kết nối trung bình trên toàn bộ đối tượng
        mean_adjacency = np.mean(graphs, axis=2)
        print(f"  FCCA input graphs: {graphs.shape[2]}, nodes: {n_nodes}")

        # Xác định tầm tìm kiếm k (số cộng đồng)
        interval_k_range = k_range_for_interval(cfg, interval_idx)
        if len(interval_k_range) == 1:
            print(f"  FCCA requested k: {interval_k_range[0]}")
        else:
            print(f"  FCCA k search: {mat2str(interval_k_range)}")

        # ===== BỨC MỘT TÍNH TOÁN CHÍNH: RUN FCCA =====
        # Áp dụng FCCA model selection để tìm cấu trúc cộng đồng tối ưu
        # Hàm này thực hiện phân tách đệ quy và đánh giá chất lượng cho từng k
        fcca = run_fcca_model_selection(graphs, interval_k_range, cfg)
        key = interval.key
        results[key] = {
            "displayName": interval.displayName,
            "frameRange": np.array([interval.frameNumbers[0], interval.frameNumbers[-1]], dtype=int),
            "frames": interval.frameNumbers.astype(int),
            "windowMs": interval.windowMs,
            "meanAdjacency": mean_adjacency,
            "labels": fcca["labels"],
            "numCommunities": len(np.unique(fcca["labels"])),
            "bestK": fcca["bestK"],
            "qualityByK": fcca["qualityByK"],
            "homogeneityByK": fcca["homogeneityByK"],
            "completenessByK": fcca["completenessByK"],
            "modularityByK": fcca["modularityByK"],
            "selectionScoreByK": fcca["selectionScoreByK"],
            "qualityTable": fcca["qualityTableArray"],
            "qualityTableColumns": np.asarray(fcca["qualityTableColumns"], dtype=object),
            "cooccurrence": fcca["cooccurrence"],
            "hierarchy": hierarchy_to_object_array(fcca["hierarchy"]),
        }

        write_matrix(csv_dir / f"{key}_mean_adjacency.csv", mean_adjacency)
        write_matrix(csv_dir / f"{key}_fcca_labels.csv", fcca["labels"].reshape(-1, 1), fmt="%d")
        write_matrix(csv_dir / f"{key}_cooccurrence.csv", fcca["cooccurrence"])
        write_quality_table(csv_dir / f"{key}_quality_by_k.csv", fcca["qualityTableRows"])

        print_quality_table(fcca["qualityTableRows"])
        print(
            "  selected k={bestK} | S={bestSelectionScore:.4f} | "
            "U={bestQuality:.4f} | H={bestHomogeneity:.4f} | "
            "C={bestCompleteness:.4f} | Q={bestModularity:.4f}\n".format(**fcca)
        )

    savemat(
        output_dir / "fcca_paper_interval_results.mat",
        {"results": matlab_ready(results)},
        do_compression=True,
    )

    plot_fcca_figures(results, output_dir, np.asarray(channel_labels, dtype=object), node_xy, cfg)

    print(f"Saved FCCA results to: {output_dir}")
    return results


def first_existing_file(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def first_existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def k_range_for_interval(cfg: Config, interval_idx: int) -> list[int]:
    if cfg.requestedK is None or is_empty_k(cfg.requestedK):
        return list(cfg.kRange)

    requested_k = np.asarray(cfg.requestedK, dtype=float).reshape(-1)
    if requested_k.size == 1:
        k = requested_k[0]
    elif requested_k.size == 3:
        k = requested_k[interval_idx - 1]
    else:
        raise ValueError("k_input must be empty, a scalar K, or a 3-element vector [preK ernK postK].")

    k = int(round(float(k)))
    if not np.isfinite(k) or k < 2:
        raise ValueError("Requested K must be an integer >= 2.")
    return [k]


def load_ho_rlsl_tensor(filename: Path) -> tuple[np.ndarray, dict[str, Any]]:
    ext = filename.suffix.lower()
    if ext == ".mat":
        return load_mat_tensor(filename)
    if ext == ".npz":
        return load_npz_tensor(filename)
    raise ValueError(f"Unsupported input extension: {ext}")


def load_mat_tensor(filename: Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta_names = [
        "filtered_change_points",
        "change_points",
        "change_point_ms",
        "raw_change_points",
        "original_shape",
    ]

    try:
        names = [name for name, _, _ in whosmat(filename)]
    except (NotImplementedError, ValueError, OSError):
        return load_hdf5_mat_tensor(filename, meta_names)

    if "low_rank" not in names:
        raise ValueError("The MAT file must contain low_rank.")

    load_names = ["low_rank"] + [name for name in meta_names if name in names]
    loaded = loadmat(filename, variable_names=load_names, squeeze_me=False, struct_as_record=False)
    tensor = np.asarray(loaded["low_rank"], dtype=np.float64)
    metadata = {
        name: np.squeeze(loaded[name])
        for name in load_names
        if name != "low_rank" and name in loaded
    }
    metadata["sourceFormat"] = "mat"
    return tensor, metadata


def load_hdf5_mat_tensor(filename: Path, meta_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "This MAT file looks like MATLAB v7.3/HDF5. Install h5py or use ho_rlsl_results.npz."
        ) from exc

    with h5py.File(filename, "r") as h5:
        if "low_rank" not in h5:
            raise ValueError("The MAT file must contain low_rank.")
        tensor = hdf5_mat_dataset_to_numpy(h5["low_rank"]).astype(np.float64)
        metadata = {name: np.squeeze(hdf5_mat_dataset_to_numpy(h5[name])) for name in meta_names if name in h5}
    metadata["sourceFormat"] = "mat"
    return tensor, metadata


def hdf5_mat_dataset_to_numpy(dataset: Any) -> np.ndarray:
    arr = np.asarray(dataset)
    if arr.ndim > 1:
        arr = np.transpose(arr, tuple(reversed(range(arr.ndim))))
    return arr


def load_npz_tensor(filename: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(filename, allow_pickle=True) as loaded:
        tensor_key = first_existing_key(loaded.files, ["low_rank", "lowrank_stream"])
        if tensor_key is None:
            available = ", ".join(loaded.files)
            raise ValueError(
                f"Cannot find low_rank or lowrank_stream inside {filename}. "
                f"Available arrays: {available}"
            )
        raw_tensor = np.asarray(loaded[tensor_key], dtype=np.float64)
        intervals = read_optional_npz(loaded, "intervals")
        tensor = normalize_npz_tensor_axes(raw_tensor, intervals)
        metadata = {
            "filtered_change_points": read_optional_npz(loaded, "filtered_change_points"),
            "change_points": read_optional_npz(loaded, "change_points"),
            "change_point_ms": read_optional_npz(loaded, "change_point_ms"),
            "intervals": intervals,
            "sourceFormat": "npz",
            "sourceTensorKey": tensor_key,
        }
        metadata = normalize_npz_index_base(metadata, tensor.shape[3])
    return tensor, metadata


def first_existing_key(keys: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def normalize_npz_tensor_axes(tensor: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if tensor.ndim != 4:
        return tensor

    frame_count = infer_frame_count_from_intervals(intervals)
    if frame_count is not None and tensor.shape[0] == frame_count and tensor.shape[1] == tensor.shape[2]:
        return np.transpose(tensor, (3, 1, 2, 0))

    if tensor.shape[1] == tensor.shape[2] and tensor.shape[0] > tensor.shape[3]:
        return np.transpose(tensor, (3, 1, 2, 0))

    return tensor


def infer_frame_count_from_intervals(intervals: np.ndarray) -> int | None:
    intervals = np.asarray(intervals)
    if intervals.size == 0:
        return None
    return int(np.max(intervals)) + (1 if int(np.min(intervals)) == 0 else 0)


def normalize_npz_index_base(metadata: dict[str, Any], n_frames: int) -> dict[str, Any]:
    intervals = np.asarray(metadata.get("intervals", np.array([])))
    is_zero_based = intervals.size > 0 and int(np.min(intervals)) == 0 and int(np.max(intervals)) == n_frames - 1
    if is_zero_based:
        for name in ("filtered_change_points", "change_points"):
            if has_nonempty_field(metadata, name):
                metadata[name] = np.asarray(metadata[name], dtype=int) + 1
        metadata["sourceIndexBase"] = 0
    else:
        metadata["sourceIndexBase"] = 1
    return metadata


def read_optional_npz(loaded: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name in loaded.files:
        return np.asarray(loaded[name])
    return np.array([])


def choose_change_points_from_metadata(
    metadata: dict[str, Any],
    n_frames: int,
    times_ms: np.ndarray,
    analysis_window_ms: tuple[float, float],
    manual_boundaries_0based: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    if manual_boundaries_0based is not None:
        return manual_boundaries_to_frame_numbers(manual_boundaries_0based, n_frames)

    if has_nonempty_field(metadata, "filtered_change_points"):
        cp = metadata["filtered_change_points"]
    elif has_nonempty_field(metadata, "change_points"):
        cp = metadata["change_points"]
    else:
        raise ValueError("ho_rlsl_results must contain filtered_change_points or change_points.")

    cp = np.rint(np.asarray(cp, dtype=float).reshape(-1)).astype(int)
    cp = stable_unique(cp[(cp > 0) & (cp < n_frames)])

    analysis_frames = np.where(
        (times_ms >= analysis_window_ms[0]) & (times_ms <= analysis_window_ms[1])
    )[0] + 1
    if analysis_frames.size == 0:
        raise ValueError(
            f"No tensor frames found in analysis window {analysis_window_ms[0]:.1f} "
            f"to {analysis_window_ms[1]:.1f} ms."
        )

    start_frame = analysis_frames[0]
    end_frame = analysis_frames[-1]
    cp = cp[(cp > start_frame) & (cp < end_frame)]
    if cp.size < 2:
        raise ValueError(
            f"Need at least two change points inside {analysis_window_ms[0]:.1f} "
            f"to {analysis_window_ms[1]:.1f} ms for Pre/ERN/Post splitting, got {mat2str(cp)}."
        )
    return np.array([cp[0], cp[-1]], dtype=int)


def manual_boundaries_to_frame_numbers(boundaries_0based: tuple[int, int, int, int], n_frames: int) -> np.ndarray:
    boundaries = np.rint(np.asarray(boundaries_0based, dtype=float).reshape(-1)).astype(int)
    if boundaries.size != 4:
        raise ValueError("Manual interval boundaries must contain exactly 4 frame numbers.")
    if np.any(boundaries < 0) or np.any(boundaries >= n_frames):
        raise ValueError(f"Manual interval boundaries must be between 0 and {n_frames - 1}.")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("Manual interval boundaries must be strictly increasing.")
    return boundaries + 1


def make_interval_specs(
    change_points: np.ndarray,
    n_frames: int,
    times_ms: np.ndarray,
    analysis_window_ms: tuple[float, float],
    pre_start_ms: float,
) -> list[IntervalSpec]:
    cp = np.rint(np.asarray(change_points, dtype=float).reshape(-1)).astype(int)
    names = ["preERN", "ERN", "postERN"]
    display_names = ["Pre-ERN", "ERN", "Post-ERN"]

    if cp.size == 4:
        if cp[0] < 1 or cp[-1] > n_frames or np.any(np.diff(cp) <= 0):
            raise ValueError(f"Invalid interval boundaries {mat2str(cp)} for {n_frames} frames.")
        frame_sets = [
            np.arange(cp[0], cp[1] + 1, dtype=int),
            np.arange(cp[1] + 1, cp[2] + 1, dtype=int),
            np.arange(cp[2] + 1, cp[3] + 1, dtype=int),
        ]
        return interval_specs_from_frame_sets(names, display_names, frame_sets, times_ms)

    if cp.size != 2:
        raise ValueError("Change points must contain either 2 points or 4 interval boundaries.")

    c1 = int(cp[0])
    c2 = int(cp[1])
    if c1 < 1 or c2 <= c1 or c2 >= n_frames:
        raise ValueError(f"Invalid change points {mat2str(change_points)} for {n_frames} frames.")

    analysis_frames = np.where(
        (times_ms >= analysis_window_ms[0]) & (times_ms <= analysis_window_ms[1])
    )[0] + 1
    if analysis_frames.size == 0:
        raise ValueError(
            f"No frames found in analysis window {analysis_window_ms[0]:.1f} "
            f"to {analysis_window_ms[1]:.1f} ms."
        )

    start_frame = analysis_frames[0]
    end_frame = analysis_frames[-1]
    pre_frames = np.where((times_ms >= pre_start_ms) & (times_ms <= times_ms[c1 - 1]))[0] + 1
    if pre_frames.size == 0:
        raise ValueError(f"No Pre-ERN frames found from {pre_start_ms:.1f} ms to change point {c1}.")
    pre_start_frame = pre_frames[0]

    if c1 < start_frame or c2 >= end_frame:
        raise ValueError(
            f"Change points {mat2str(change_points)} must lie inside analysis frame range "
            f"{start_frame}-{end_frame} ({analysis_window_ms[0]:.1f} to {analysis_window_ms[1]:.1f} ms)."
        )

    frame_sets = [
        np.arange(pre_start_frame, c1 + 1, dtype=int),
        np.arange(c1 + 1, c2 + 1, dtype=int),
        np.arange(c2 + 1, end_frame + 1, dtype=int),
    ]

    intervals = []
    for name, display_name, frames in zip(names, display_names, frame_sets):
        intervals.append(
            IntervalSpec(
                key=name,
                displayName=display_name,
                indices=frames - 1,
                frameNumbers=frames,
                windowMs=np.array([times_ms[frames[0] - 1], times_ms[frames[-1] - 1]], dtype=float),
            )
        )
    return intervals


def interval_specs_from_frame_sets(
    names: list[str],
    display_names: list[str],
    frame_sets: list[np.ndarray],
    times_ms: np.ndarray,
) -> list[IntervalSpec]:
    intervals = []
    for name, display_name, frames in zip(names, display_names, frame_sets):
        intervals.append(
            IntervalSpec(
                key=name,
                displayName=display_name,
                indices=frames - 1,
                frameNumbers=frames,
                windowMs=np.array([times_ms[frames[0] - 1], times_ms[frames[-1] - 1]], dtype=float),
            )
        )
    return intervals


def interval_graphs_from_tensor(tensor: np.ndarray, frame_indices: np.ndarray, cfg: Config) -> np.ndarray:
    n_subjects, n_nodes, _, _ = tensor.shape
    graphs = np.zeros((n_nodes, n_nodes, n_subjects), dtype=np.float32)

    for subject_idx in range(n_subjects):
        subject_tensor = tensor[subject_idx]
        W = np.mean(np.take(subject_tensor, frame_indices, axis=2), axis=2)
        graphs[:, :, subject_idx] = prepare_adjacency(W).astype(np.float32)

    return graphs


def prepare_adjacency(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64).copy()
    W[~np.isfinite(W)] = 0.0
    W = (W + W.T) / 2.0
    W[W < 0] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def run_fcca_model_selection(graphs: np.ndarray, k_range: list[int], cfg: Config) -> dict[str, Any]:
    """
    Thực hiện phân tách FCCA đệ quy để tìm cấu trúc cộng đồng tối ưu.
    
    Thuật toán:
    1. Tính toán rank bins từ tất cả đồ thị (đã chuẩn bị sẵn)
    2. Khởi tạo với 1 cụm chứa toàn bộ nút
    3. Lặp từ k=2 đến k_max:
       - Chọn cụm có điểm số thấp nhất (cần phân tách)
       - Phân tách nó thành 2 cộng đồng bằng Fiedler consensus
       - Tính chất lượng FCCA (U), homogeneity (H), completeness (C), modularity (Q)
       - Lưu lại cấu trúc phân cụm
    4. Chọn k tối ưu dựa trên tiêu chí selection (U hoặc U×Q)
    5. Trả về kết quả cho k tốt nhất
    """
    n = graphs.shape[0]
    k_max = int(max(k_range))
    # Tính toán rank bins: ma trận xếp hạng cho mỗi cạnh trong tất cả đồ thị
    # Dùng để tính toán entropy và các đại lượng chất lượng
    rank_bins = compute_rank_bins(graphs)

    # ===== KHỞI TẠO =====
    clusters = [np.arange(n, dtype=int)]  # Bắt đầu với 1 cụm chứa tất cả nút
    hierarchy: list[list[np.ndarray] | None] = [None] * (k_max + 1)  # Lưu cấu trúc ở mỗi k
    quality_by_k = np.full(k_max + 1, np.nan)  # FCCA quality measure (U)
    homogeneity_by_k = np.full(k_max + 1, np.nan)  # Độ thuần nhất (H)
    completeness_by_k = np.full(k_max + 1, np.nan)  # Độ đầy đủ (C)
    modularity_by_k = np.full(k_max + 1, np.nan)  # Modularity (Q)
    cooccur_by_k: list[np.ndarray | None] = [None] * (k_max + 1)  # Ma trận đồng xuất hiện

    # ===== VÒNG LẶP PHÂN TÁCH ĐỆ QUY =====
    for k in range(2, k_max + 1):
        # Bước 1: Tìm cụm có "điểm số xấu" nhất (cần phân tách nhất)
        # Điểm số = weighted combination of homogeneity & completeness
        split_idx = choose_cluster_to_split(graphs, rank_bins, clusters, cfg)
        if split_idx is None:
            break  # Không có cụm nào có thể phân tách được

        # Bước 2: Phân tách cụm được chọn thành 2 phần
        nodes = clusters[split_idx]
        part_a, part_b, cooccur = fcca_split_cluster(graphs, nodes)
        # Kiểm tra xem phân tách có hợp lệ không (size ≥ minCommunitySize)
        if (
            part_a.size == 0
            or part_b.size == 0
            or part_a.size < cfg.minCommunitySize
            or part_b.size < cfg.minCommunitySize
        ):
            break  # Phân tách không thỏa điều kiện, dừng

        # Bước 3: Cập nhật tập cụm
        clusters.pop(split_idx)  # Xóa cụm cũ
        clusters.extend([part_a, part_b])  # Thêm 2 cụm mới
        clusters = sort_clusters_by_first_node(clusters)  # Sắp xếp theo nút đầu tiên

        # Bước 4: Tính toán chất lượng cho cấu trúc k-cụm này
        labels = clusters_to_labels(clusters, n)  # Chuyển cụm → nhãn nút
        # Tính chất lượng FCCA, homogeneity, completeness dựa trên rank distributions
        U, Hbar, Cbar, _, _ = fcca_quality(rank_bins, labels)
        # Tính modularity từ ma trận kết nối trung bình
        Q = weighted_modularity(np.mean(graphs, axis=2), labels)
        # Lưu kết quả
        hierarchy[k] = [cluster.copy() for cluster in clusters]  # Lưu cấu trúc ở k này
        quality_by_k[k] = U
        homogeneity_by_k[k] = Hbar
        completeness_by_k[k] = Cbar
        modularity_by_k[k] = Q

        # Lưu ma trận đồng xuất hiện toàn cộng (đã phân tách cụm này)
        cooccur_full = np.zeros((n, n), dtype=float)
        cooccur_full[np.ix_(nodes, nodes)] = cooccur
        cooccur_by_k[k] = cooccur_full

    # ===== CHỌN K TỐI ƯU =====
    # Lọc các k có kết quả hợp lệ (không phải NaN)
    valid_k = np.array([k for k in k_range if np.isfinite(quality_by_k[k])], dtype=int)
    if valid_k.size == 0:
        raise RuntimeError(f"FCCA failed to produce any valid k in {mat2str(k_range)}.")

    # Tính selection score dựa trên tiêu chí (default: U, hoặc U×positive Q)
    selection_score_by_k = model_selection_score(quality_by_k, modularity_by_k, valid_k, cfg)
    # Chọn k có selection score cao nhất
    best_pos = int(np.nanargmax(selection_score_by_k[valid_k]))
    best_k = int(valid_k[best_pos])
    best_clusters = hierarchy[best_k]
    if best_clusters is None:
        raise RuntimeError(f"FCCA produced no hierarchy for k={best_k}.")
    labels = clusters_to_labels(best_clusters, n)

    quality_columns = ["k", "SelectionScore", "U", "H", "C", "Q"]
    quality_rows = [
        {
            "k": int(k),
            "SelectionScore": float(selection_score_by_k[k]),
            "U": float(quality_by_k[k]),
            "H": float(homogeneity_by_k[k]),
            "C": float(completeness_by_k[k]),
            "Q": float(modularity_by_k[k]),
        }
        for k in valid_k
    ]
    quality_array = np.array([[row[column] for column in quality_columns] for row in quality_rows], dtype=float)

    return {
        "bestK": best_k,
        "labels": labels,
        "bestQuality": float(quality_by_k[best_k]),
        "bestHomogeneity": float(homogeneity_by_k[best_k]),
        "bestCompleteness": float(completeness_by_k[best_k]),
        "bestModularity": float(modularity_by_k[best_k]),
        "bestSelectionScore": float(selection_score_by_k[best_k]),
        "qualityByK": quality_by_k[1:],
        "homogeneityByK": homogeneity_by_k[1:],
        "completenessByK": completeness_by_k[1:],
        "modularityByK": modularity_by_k[1:],
        "selectionScoreByK": selection_score_by_k[1:],
        "qualityTableRows": quality_rows,
        "qualityTableArray": quality_array,
        "qualityTableColumns": quality_columns,
        "cooccurrence": cooccur_by_k[best_k],
        "hierarchy": hierarchy,
    }


def model_selection_score(
    quality_by_k: np.ndarray,
    modularity_by_k: np.ndarray,
    valid_k: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    selection_score_by_k = np.full_like(quality_by_k, np.nan, dtype=float)
    if cfg.selectionCriterion == "U":
        selection_score_by_k[valid_k] = quality_by_k[valid_k]
    elif cfg.selectionCriterion == "U_times_positive_Q":
        q = modularity_by_k[valid_k].copy()
        q[(q < 0) | (~np.isfinite(q))] = 0
        if np.all(q <= 0):
            selection_score_by_k[valid_k] = quality_by_k[valid_k]
        else:
            selection_score_by_k[valid_k] = quality_by_k[valid_k] * q
    else:
        raise ValueError(f"Unknown selection criterion: {cfg.selectionCriterion}")
    return selection_score_by_k


def sort_clusters_by_first_node(clusters: list[np.ndarray]) -> list[np.ndarray]:
    return [clusters[idx] for idx in np.argsort([np.min(cluster) for cluster in clusters])]


def choose_cluster_to_split(
    graphs: np.ndarray,
    rank_bins: np.ndarray,
    clusters: list[np.ndarray],
    cfg: Config,
) -> int | None:
    """
    Lựa chọn cụm "xấu nhất" cần phân tách dựa trên điểm số chất lượng.
    
    Phương pháp:
    1. Tính chất lượng FCCA (homogeneity & completeness) cho từng cụm
    2. Với mỗi giá trị γ ∈ [0, 0.1, 0.2, ..., 1]:
       - Tính điểm số ζ = C×γ + H×(1-γ) cho từng cụm
       - Chọn cụm có ζ nhỏ nhất
    3. Tính số lần mỗi cụm được chọn (tổng trên tất cả γ)
    4. Trả về cụm được chọn nhiều nhất (nó cần phân tách nhất)
    """
    # Trường hợp chỉ có 1 cụm
    if len(clusters) == 1:
        return 0 if clusters[0].size >= 2 * cfg.minCommunitySize else None

    n_clusters = len(clusters)
    counts = np.zeros(n_clusters, dtype=float)  # Đếm số lần được chọn cho mỗi cụm
    labels = clusters_to_labels(clusters, graphs.shape[0])
    # Tính homogeneity & completeness cho từng cụm
    _, _, _, cluster_h, cluster_c = fcca_quality(rank_bins, labels)

    # Thử các trọng số khác nhau: γ ∈ {0, 0.1, 0.2, ..., 1}
    # Mục đích: chọn cụm xấu nhất dựa trên tổng hợp nhiều tiêu chí
    for gamma in np.arange(0.0, 1.0 + 1e-12, 0.1):
        zeta = np.full(n_clusters, np.inf)  # Điểm số cho mỗi cụm
        for idx, cluster in enumerate(clusters):
            # Chỉ xét cụm có thể phân tách (size ≥ minCommunitySize)
            if cluster.size >= 2 * cfg.minCommunitySize:
                # Điểm số = combination của completeness & homogeneity
                zeta[idx] = cluster_c[idx] * gamma + cluster_h[idx] * (1.0 - gamma)
        # Chọn cụm có điểm số thấp nhất (xấu nhất)
        bad_idx = int(np.argmin(zeta))
        counts[bad_idx] += 1  # Tăng bộ đếm

    # Đánh dấu cụm không thể phân tách (quá nhỏ)
    for idx, cluster in enumerate(clusters):
        if cluster.size < 2 * cfg.minCommunitySize:
            counts[idx] = -np.inf  # Không được lựa chọn

    # Kiểm tra xem có cụm nào có thể phân tách không
    if np.all(~np.isfinite(counts)):
        return None  # Không có cụm nào phù hợp
    # Trả về cụm được chọn nhiều nhất trên tất cả γ
    return int(np.argmax(counts))


def fcca_split_cluster(graphs: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phân tách một cụm thành 2 phần bằng Fiedler Consensus Clustering.
    
    Quá trình:
    1. Với mỗi đồ thị từ từng đối tượng (m đồ thị):
       - Trích xuất sub-graph từ các nút của cụm hiện tại
       - Áp dụng Fiedler bipartitioning để tách thành 2 phần
       - Xây dựng ma trận đồng xuất hiện cục bộ
    2. Chuẩn hóa: chia cho m để được xác suất
    3. Áp dụng Fiedler một lần nữa trên ma trận đồng xuất hiện
       để phát hiện cấu trúc phân hoạch đồng thuận
    4. Trả về 2 phần được phân hoạch
    """
    m = graphs.shape[2]  # Số đối tượng
    p = nodes.size  # Số nút trong cụm hiện tại
    cooccur = np.zeros((p, p), dtype=float)  # Ma trận đồng xuất hiện cục bộ

    # Lặp qua tất cả m đồ thị từ m đối tượng
    for graph_idx in range(m):
        # Trích xuất sub-graph chứa các nút của cụm này
        W = graphs[:, :, graph_idx][np.ix_(nodes, nodes)]
        # Áp dụng Fiedler bipartition để phân tách
        local_labels = fiedler_bipartition(W)
        # Cập nhật ma trận đồng xuất hiện
        cooccur += (local_labels[:, None] == local_labels[None, :]).astype(float)

    # Chuẩn hóa: chia cho m để được xác suất đồng xuất hiện
    cooccur /= m
    # Áp dụng Fiedler consensus: tìm phân hoạch từ ma trận đồng xuất hiện
    consensus_labels = fiedler_bipartition(cooccur)
    # Chia nút thành 2 phần dựa trên consensus label
    part_a = nodes[consensus_labels == 1]
    part_b = nodes[consensus_labels == 2]
    return part_a, part_b, cooccur


def fiedler_bipartition(W: np.ndarray) -> np.ndarray:
    """
    Phân hoạch nhị phân một đồ thị dựa trên Fiedler vector.
    
    Quá trình:
    1. Xây dựng ma trận Laplacian chuẩn hóa đối xứng
    2. Tính vector riêng ứng với trị riêng nhỏ nhất khác không (Fiedler vector)
    3. Sắp xếp Fiedler vector theo thứ tự tăng dần
    4. Tìm vị trí có khe hở lớn nhất (maximum gap)
    5. Phân tách tại vị trí đó thành 2 cộng đồng
    """
    n = W.shape[0]
    # Trường hợp đặc biệt
    if n == 1:
        return np.array([1], dtype=int)
    if n == 2:
        return np.array([1, 2], dtype=int)

    # Chuẩn bị ma trận kề (đối xứng, không có self-loop, giá trị dương)
    W = prepare_adjacency(W)
    # Tính bậc (degree) từ ma trận kề
    degree = np.sum(W, axis=1)
    # Kiểm tra nút nào có bậc > 0 (nút hợp lệ)
    valid = degree > EPS
    if np.count_nonzero(valid) < 2:
        # Nếu quá ít nút hợp lệ, dùng phân tách 50-50
        return fallback_half_split(n)

    # Tính ma trận Laplacian chuẩn hóa đối xứng
    # L = D^(-1/2) * (D - W) * D^(-1/2)
    inv_sqrt_degree = np.zeros(n, dtype=float)
    inv_sqrt_degree[valid] = 1.0 / np.sqrt(degree[valid])
    L = np.eye(n) - (inv_sqrt_degree[:, None] * W) * inv_sqrt_degree[None, :]
    # Đảm bảo ma trận đối xứng
    L = (L + L.T) / 2.0

    # Tính tất cả trị riêng & vector riêng của Laplacian
    evals, V = np.linalg.eigh(L)
    # Sắp xếp theo thứ tự trị riêng tăng dần
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    V = np.real(V[:, order])
    # Tìm các trị riêng khác không (threshold: 1e-10)
    # Trị riêng đầu tiên (index 0) sẽ ≈ 0 (trivial), ta cần trị riêng tiếp theo
    fiedler_candidates = np.where(evals > 1e-10)[0]
    if fiedler_candidates.size == 0:
        return fallback_half_split(n)

    # Lấy Fiedler vector (vector riêng ứng với trị riêng nhỏ nhất khác không)
    u = V[:, fiedler_candidates[0]]
    # Sắp xếp các phần tử của Fiedler vector theo thứ tự tăng dần
    sort_order = np.argsort(u, kind="mergesort")
    u_sorted = u[sort_order]
    # Tính các khe hở (gaps) giữa các phần tử liên tiếp
    gaps = np.diff(u_sorted)
    if gaps.size == 0:
        return fallback_half_split(n)

    # Tìm vị trí có khe hở lớn nhất
    gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[gap_idx])
    # Nếu khe hở quá nhỏ, không có cấu trúc phân tách rõ ràng
    if max_gap <= EPS:
        return fallback_half_split(n)

    # Tạo nhãn: nút trước khe hở → 1, sau khe hở → 2
    labels = np.zeros(n, dtype=int)
    split = gap_idx + 1
    labels[sort_order[:split]] = 1
    labels[sort_order[split:]] = 2
    # Kiểm tra xem phân hoạch có hợp lệ (có 2 phần)
    if np.unique(labels).size < 2:
        return fallback_half_split(n)
    return labels


def fallback_half_split(n: int) -> np.ndarray:
    labels = np.ones(n, dtype=int)
    labels[n // 2 :] = 2
    return labels


def clusters_to_labels(clusters: list[np.ndarray], n: int) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    for idx, cluster in enumerate(clusters, start=1):
        labels[cluster] = idx
    return relabel(labels)


def relabel(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    unique_labels = stable_unique(labels)
    out = np.zeros_like(labels)
    for idx, label in enumerate(unique_labels, start=1):
        out[labels == label] = idx
    return out


def cooccurrence_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    return (labels[:, None] == labels[None, :]).astype(float)


def compute_rank_bins(graphs: np.ndarray) -> np.ndarray:
    n, _, m = graphs.shape
    rank_bins = np.zeros((n, n, m), dtype=np.uint16)

    for graph_idx in range(m):
        W = prepare_adjacency(graphs[:, :, graph_idx])
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


def fcca_quality(rank_bins: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Tính toán chất lượng FCCA cho một cấu trúc phân cụm.
    
    Đầu vào:
    - rank_bins: ma trận xếp hạng (nodes × nodes × graphs)
    - labels: nhãn cụm cho từng nút
    
    Đầu ra:
    - U: FCCA quality (Harmonic mean của homogeneity & completeness)
    - Hbar: trung bình homogeneity trên tất cả cụm
    - Cbar: trung bình completeness trên tất cả cụm
    - cluster_h, cluster_c: homogeneity & completeness từng cụm
    """
    labels = relabel(labels)  # Chuẩn hóa nhãn
    n = labels.size
    k = int(np.max(labels))
    cluster_h = np.zeros(k, dtype=float)  # Homogeneity cho từng cụm
    cluster_c = np.zeros(k, dtype=float)  # Completeness cho từng cụm

    # Tính homogeneity & completeness cho mỗi cụm
    for c in range(1, k + 1):
        # Lấy rank distributions của cạnh bên trong & bên ngoài cụm
        pintra, pinter = rank_distributions_for_cluster(rank_bins, labels, c, n)
        # Entropy của rank distribution bên trong cụm
        entropy_intra = entropy_base2(pintra)
        # Tỷ lệ nút trong cụm
        alpha = np.sum(labels == c) / n
        # Homogeneity = α × (1 - entropy_intra / log n)
        # Phản ánh mức độ các nút trong cụm có rank thấp với nhau
        cluster_h[c - 1] = alpha * max(0.0, 1.0 - entropy_intra / max(np.log2(n), EPS))
        # Completeness = Jensen-Shannon distance giữa in-cluster và inter-cluster rank dist
        cluster_c[c - 1] = jensen_shannon(pintra, pinter)

    # Trung bình homogeneity & completeness trên toàn bộ cụm
    Hbar = float(np.mean(cluster_h))
    Cbar = float(np.mean(cluster_c))
    # FCCA quality = Harmonic mean(Hbar, Cbar) = 2 / (1/Hbar + 1/Cbar)
    if Hbar <= 0 or Cbar <= 0:
        U = 0.0
    else:
        U = 2.0 / (1.0 / Hbar + 1.0 / Cbar)  # Harmonic mean
    return U, Hbar, Cbar, cluster_h, cluster_c


def rank_distributions_for_cluster(
    rank_bins: np.ndarray,
    labels: np.ndarray,
    c: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def histcounts_discrete(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.rint(np.asarray(values)).astype(int).reshape(-1)
    values = values[(values >= 1) & (values <= n_bins)]
    if values.size == 0:
        return np.zeros(n_bins, dtype=float)
    return np.bincount(values, minlength=n_bins + 1)[1 : n_bins + 1].astype(float)


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = np.sum(counts)
    if total <= 0:
        return np.ones_like(counts, dtype=float) / counts.size
    return counts / total


def entropy_base2(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    js = 0.5 * kl_base2(p, m) + 0.5 * kl_base2(q, m)
    return float(max(0.0, min(1.0, js)))


def kl_base2(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def weighted_modularity(A: np.ndarray, labels: np.ndarray) -> float:
    A = prepare_adjacency(A)
    labels = np.asarray(labels).reshape(-1)
    m2 = np.sum(A)
    if m2 <= EPS:
        return 0.0

    degree = np.sum(A, axis=1)
    same_cluster = labels[:, None] == labels[None, :]
    expected = np.outer(degree, degree) / m2
    q = np.sum((A - expected) * same_cluster) / m2
    return float(q)


def plot_fcca_figures(
    results: dict[str, Any],
    output_dir: Path,
    channel_labels: np.ndarray,
    node_xy: np.ndarray,
    cfg: Config,
) -> None:
    interval_keys = ["preERN", "ERN", "postERN"]
    before_fig, before_axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor="white", constrained_layout=True)
    after_fig, after_axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor="white", constrained_layout=True)
    matrix_before_fig, matrix_before_axes = plt.subplots(
        1, 3, figsize=(20.48, 7.2), facecolor="white", constrained_layout=True
    )
    matrix_after_fig, matrix_after_axes = plt.subplots(
        1, 3, figsize=(20.48, 7.2), facecolor="white", constrained_layout=True
    )

    for idx, key in enumerate(interval_keys):
        R = results[key]
        A = R["meanAdjacency"]
        labels = R["labels"]
        order = np.argsort(labels, kind="mergesort")

        plot_head_network(
            A,
            None,
            node_xy,
            channel_labels,
            f"{R['displayName']} input connectivity",
            cfg,
            before_axes[idx],
        )
        plot_head_network(
            A,
            labels,
            node_xy,
            channel_labels,
            f"{R['displayName']} FCCA communities ({R['bestK']})",
            cfg,
            after_axes[idx],
        )
        plot_connectivity_matrix(
            A,
            channel_labels,
            f"{R['displayName']} input matrix",
            None,
            matrix_before_axes[idx],
        )
        plot_connectivity_matrix(
            A[np.ix_(order, order)],
            channel_labels[order],
            f"{R['displayName']} FCCA-sorted matrix",
            labels[order],
            matrix_after_axes[idx],
        )

    before_fig.suptitle("Before FCCA: Mean Input Brain Connectivity Networks", fontweight="bold", fontsize=18)
    after_fig.suptitle("After FCCA: Fiedler Consensus Communities", fontweight="bold", fontsize=18)
    matrix_before_fig.suptitle("Before FCCA: Mean Input Connectivity Matrices", fontweight="bold", fontsize=18)
    matrix_after_fig.suptitle("After FCCA: Connectivity Matrices Sorted by Community", fontweight="bold", fontsize=18)

    save_figure(before_fig, output_dir / "brain_networks_before_fcca.png")
    save_figure(after_fig, output_dir / "brain_networks_after_fcca.png")
    save_figure(matrix_before_fig, output_dir / "connectivity_matrices_before_fcca.png")
    save_figure(matrix_after_fig, output_dir / "connectivity_matrices_after_fcca.png")


def plot_connectivity_matrix(
    A: np.ndarray,
    labels_text: np.ndarray,
    title_text: str,
    community_labels: np.ndarray | None,
    ax: plt.Axes,
) -> None:
    image = ax.imshow(A, aspect="equal", cmap=matrix_colormap())
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title_text, fontweight="bold", fontsize=10)
    n = A.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([str(label) for label in labels_text], fontsize=7, rotation=90)
    ax.set_yticklabels([str(label) for label in labels_text], fontsize=7)
    ax.tick_params(direction="in")
    for spine in ax.spines.values():
        spine.set_visible(True)

    if community_labels is not None:
        boundaries = np.where(np.diff(np.asarray(community_labels).reshape(-1)) != 0)[0] + 0.5
        for boundary in boundaries:
            ax.axvline(boundary, color="white", linewidth=1.1)
            ax.axhline(boundary, color="white", linewidth=1.1)


def plot_head_network(
    A: np.ndarray,
    labels: np.ndarray | None,
    xy: np.ndarray,
    channel_labels: np.ndarray,
    title_text: str,
    cfg: Config,
    ax: plt.Axes,
) -> None:
    n = A.shape[0]
    A = prepare_adjacency(A)
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
    n_edges = min(cfg.maxPlotEdges, weight.size)

    ax.set_aspect("equal")
    ax.axis("off")
    draw_scalp_outline(ax)

    if n_edges > 0:
        selected = weight[:n_edges]
        weight_range = np.max(selected) - np.min(selected)
        if weight_range <= EPS:
            scaled = np.ones_like(selected)
        else:
            scaled = (selected - np.min(selected)) / weight_range

        for edge_idx in range(n_edges - 1, -1, -1):
            shade = 0.18 + 0.55 * (1.0 - scaled[edge_idx])
            ax.plot(
                xy[[row[edge_idx], col[edge_idx]], 0],
                xy[[row[edge_idx], col[edge_idx]], 1],
                color=(shade, shade, shade),
                linewidth=0.35 + 1.7 * scaled[edge_idx],
            )

    if labels is None:
        node_colors = np.tile(np.array([0.00, 0.70, 0.78]), (n, 1))
    else:
        labels = np.asarray(labels, dtype=int).reshape(-1)
        palette = community_colors(int(np.max(labels)))
        node_colors = palette[labels - 1, :]
        isolated_nodes = nodes_without_positive_intra_cluster_edge(A, labels)
        node_colors[isolated_nodes, :] = np.tile(
            np.asarray(cfg.isolatedNodeColor), (isolated_nodes.size, 1)
        )

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=58,
        c=node_colors,
        edgecolors=(0.12, 0.12, 0.12),
        linewidths=0.4,
        zorder=3,
    )

    for node in range(n):
        direction = xy[node, :].copy()
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 0.15:
            direction = np.array([0.0, 1.0])
            direction_norm = 1.0
        label_xy = xy[node, :] + 0.09 * direction / direction_norm
        ax.text(
            label_xy[0],
            label_xy[1],
            str(channel_labels[node]),
            fontsize=6.5,
            fontweight="bold",
            horizontalalignment="center",
            color=(0.05, 0.05, 0.05),
        )

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.28, 1.22)
    ax.set_title(title_text, fontweight="bold", fontsize=11)


def nodes_without_positive_intra_cluster_edge(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = relabel(labels)
    n = labels.size
    isolated_nodes = []
    all_nodes = np.arange(n)

    for node in range(n):
        same_cluster = all_nodes[(labels == labels[node]) & (all_nodes != node)]
        if same_cluster.size == 0:
            isolated_nodes.append(node)
            continue
        if np.nansum(A[node, same_cluster]) <= EPS:
            isolated_nodes.append(node)

    return np.asarray(isolated_nodes, dtype=int)


def draw_scalp_outline(ax: plt.Axes) -> None:
    theta = np.linspace(0, 2.0 * np.pi, 240)
    ax.fill(
        1.03 * np.cos(theta),
        1.03 * np.sin(theta),
        color=(0.985, 0.985, 0.985),
        edgecolor=(0.35, 0.35, 0.35),
        linewidth=0.9,
        zorder=0,
    )
    ax.plot([0, -0.08, 0.08, 0], [1.03, 1.16, 1.03, 1.03], color=(0.35, 0.35, 0.35), linewidth=0.9)
    ax.plot(
        [-1.03, -1.13, -1.13, -1.03],
        [0.22, 0.12, -0.12, -0.22],
        color=(0.35, 0.35, 0.35),
        linewidth=0.9,
    )
    ax.plot(
        [1.03, 1.13, 1.13, 1.03],
        [0.22, 0.12, -0.12, -0.22],
        color=(0.35, 0.35, 0.35),
        linewidth=0.9,
    )


def ern_head_layout(n_nodes: int, raw_bids_dir: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    if raw_bids_dir is not None and raw_bids_dir.is_dir():
        files = sorted(raw_bids_dir.glob("sub-*/eeg/*_electrodes.tsv"))
        if files:
            path = files[0]
            try:
                names, x, y, z = read_electrodes_tsv(path)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                idx = np.where(valid)[0]
                if idx.size >= n_nodes:
                    keep = idx[:n_nodes]
                    labels = names[keep]
                    xy = np.column_stack([-y[keep], x[keep]])
                    radius = np.max(np.sqrt(np.sum(xy**2, axis=1)))
                    if radius > EPS:
                        xy = 0.96 * xy / radius
                    note = f"Scalp locations read from {path.name}."
                    return xy, labels, note
            except Exception as exc:
                warnings.warn(f"Could not read electrode layout: {exc}", RuntimeWarning)

    theta = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, n_nodes + 1)[:-1]
    xy = np.column_stack([0.88 * np.cos(theta), 0.88 * np.sin(theta)])
    labels = np.asarray([f"N{idx}" for idx in range(1, n_nodes + 1)], dtype=object)
    note = "No electrode layout supplied; nodes are displayed on a circle."
    return xy, labels, note


def read_electrodes_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = []
    x_values = []
    y_values = []
    z_values = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            names.append(row.get("name", ""))
            x_values.append(numeric_value(row.get("x")))
            y_values.append(numeric_value(row.get("y")))
            z_values.append(numeric_value(row.get("z")))

    return (
        np.asarray(names, dtype=object),
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(z_values, dtype=float),
    )


def numeric_value(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "n/a", "nan", "na"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def save_figure(fig: plt.Figure, filename: Path) -> None:
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_matrix(filename: Path, values: np.ndarray, fmt: str = "%.10g") -> None:
    np.savetxt(filename, np.asarray(values), delimiter=",", fmt=fmt)


def write_quality_table(filename: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["k", "SelectionScore", "U", "H", "C", "Q"]
    with filename.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_quality_table(rows: list[dict[str, float]]) -> None:
    fieldnames = ["k", "SelectionScore", "U", "H", "C", "Q"]
    print("  " + "  ".join(f"{name:>14}" for name in fieldnames))
    for row in rows:
        print(
            "  "
            + "  ".join(
                f"{int(row[name]):14d}" if name == "k" else f"{float(row[name]):14.6f}"
                for name in fieldnames
            )
        )


def matrix_colormap() -> str:
    return "parula" if "parula" in plt.colormaps() else "viridis"


def community_colors(n_labels: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(n_labels, 1))
    return np.asarray(cmap(np.arange(n_labels)))[:, :3]


def has_nonempty_field(metadata: dict[str, Any], field: str) -> bool:
    if field not in metadata:
        return False
    value = metadata[field]
    if value is None:
        return False
    return np.asarray(value).size > 0


def stable_unique(values: np.ndarray | list[Any]) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    seen = set()
    out = []
    for value in arr:
        key = value.item() if hasattr(value, "item") else value
        if key not in seen:
            seen.add(key)
            out.append(value)
    return np.asarray(out, dtype=arr.dtype if arr.size else int)


def is_empty_k(k_input: Any) -> bool:
    try:
        return np.asarray(k_input).size == 0
    except Exception:
        return False


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return {
        "sampleRateHz": cfg.sampleRateHz,
        "epochStartMs": cfg.epochStartMs,
        "analysisWindowMs": np.asarray(cfg.analysisWindowMs, dtype=float),
        "preStartMs": cfg.preStartMs,
        "kRange": np.asarray(cfg.kRange, dtype=int),
        "selectionCriterion": cfg.selectionCriterion,
        "requestedK": np.asarray([] if cfg.requestedK is None else cfg.requestedK),
        "minCommunitySize": cfg.minCommunitySize,
        "maxPlotEdges": cfg.maxPlotEdges,
        "isolatedNodeColor": np.asarray(cfg.isolatedNodeColor, dtype=float),
        "maxGraphsPerInterval": cfg.maxGraphsPerInterval,
        "randomSeed": cfg.randomSeed,
    }


def interval_to_result(interval: IntervalSpec) -> dict[str, Any]:
    return {
        "key": interval.key,
        "displayName": interval.displayName,
        "indices": interval.frameNumbers.astype(int),
        "windowMs": interval.windowMs,
    }


def hierarchy_to_object_array(hierarchy: list[list[np.ndarray] | None]) -> np.ndarray:
    cells = np.empty(len(hierarchy), dtype=object)
    for idx, clusters in enumerate(hierarchy):
        if clusters is None:
            cells[idx] = np.array([], dtype=object)
        else:
            cluster_cells = np.empty(len(clusters), dtype=object)
            for cluster_idx, cluster in enumerate(clusters):
                cluster_cells[cluster_idx] = np.asarray(cluster, dtype=int) + 1
            cells[idx] = cluster_cells
    return cells


def matlab_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: matlab_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        arr = np.empty(len(value), dtype=object)
        for idx, item in enumerate(value):
            arr[idx] = matlab_ready(item)
        return arr
    if isinstance(value, tuple):
        return np.asarray(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return value
    return value


def mat2str(values: Any, precision: int | None = None) -> str:
    arr = np.asarray(values).reshape(-1)
    parts = []
    for value in arr:
        if precision is None:
            value_float = float(value)
            if np.isfinite(value_float) and value_float.is_integer():
                parts.append(str(int(value_float)))
            else:
                parts.append(f"{value_float:g}")
        else:
            parts.append(f"{float(value):.{precision}g}")
    return "[" + " ".join(parts) + "]"


def parse_k_input(argv: list[str]) -> Any:
    if not argv:
        return None

    text = " ".join(argv).strip()
    if text in {"", "[]"}:
        return None

    for char in "[](),;":
        text = text.replace(char, " ")
    parts = [part for part in text.split() if part]
    if not parts:
        return None

    values = [int(round(float(part))) for part in parts]
    if len(values) == 1:
        return values[0]
    return values


if __name__ == "__main__":
    cli_k_input = parse_k_input(sys.argv[1:])
    cluster_ern_fcca_paper(DEFAULT_K_INPUT if cli_k_input is None else cli_k_input)
