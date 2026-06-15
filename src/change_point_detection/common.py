import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from preprocessing.config_1024 import config as prep_config

# Setup matplotlib styling
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.autolayout": True
})

def load_tensor(file_path):
    """
    Convert tensor shape from:
    (Subjects, Nodes, Nodes, Time)
    ->
    (Nodes, Nodes, Subjects, Time)
    """
    return np.transpose(np.load(file_path), (1, 2, 0, 3))

def get_project_paths():
    """Resolve project directories relative to this file."""
    ROOT = Path(__file__).resolve().parent.parent.parent
    
    # Preprocessing, tracking, and FCCA results always go to outputs/
    data_dir = ROOT / "outputs"
        
    return {
        "ROOT": ROOT,
        "DATA": data_dir,
        "HOSVD_OUT": data_dir / "fcca_results/HOSVD",
        "HORLSL_OUT": data_dir / "fcca_results/HO_RLSL",
        "PELT_OUT": data_dir / "fcca_results/PELT",
        "CP_OUT": data_dir / "fcca_results/CP_Tracking",
        "DMD_OUT": data_dir / "fcca_results/DMD",
        "PLOTS": data_dir / "fcca_results/Plots"
    }

def load_available_tensors():
    """Load standard 1024Hz tensors from the project root data/outputs directory."""
    paths = get_project_paths()
    ROOT = paths["ROOT"]
    
    # Check possible locations
    locations = [
        ROOT / "outputs" / "tensor_4d_1024",
        ROOT / "outputs" / "processed_1024/03_connectivity_tensors",
        ROOT / "data" / "tensor_4d_1024",
        ROOT / "data" / "processed_1024/03_connectivity_tensors",
    ]
    
    inc_1024_path = None
    cor_1024_path = None
    
    for loc in locations:
        inc_p = loc / "tensor_incorrect_4d.npy"
        cor_p = loc / "tensor_correct_4d.npy"
        if inc_p.exists() and cor_p.exists():
            inc_1024_path = inc_p
            cor_1024_path = cor_p
            break
            
    if inc_1024_path is None:
        raise FileNotFoundError("Tensors not found. Run preprocessing first or place them under outputs/tensor_4d_1024/ or data/tensor_4d_1024/")
        
    print(f"Loading tensors from {inc_1024_path.parent}")
    tensor_ern_1024 = load_tensor(inc_1024_path)
    tensor_crn_1024 = load_tensor(cor_1024_path)
    
    T_1024 = tensor_ern_1024.shape[-1]
    tmin_ms = prep_config.eeg.TMIN * 1000
    tmax_ms = prep_config.eeg.TMAX * 1000
    time_1024 = np.linspace(tmin_ms, tmax_ms, T_1024)
    
    return {
        "1024Hz_ERN": (tensor_ern_1024, time_1024),
        "1024Hz_CRN": (tensor_crn_1024, time_1024),
    }

def extract_physiological_cps(signal: np.ndarray, time_ms: np.ndarray, n_bkps: int = 4) -> list:
    """
    Extracts Change Points with physiological constraints:
    Fades edges to ignore boundary artifacts and enforces minimum state duration.
    """
    T = len(signal)
    signal_2d = signal.reshape(-1, 1)
    
    # Apply fade window (10% at edges)
    window = np.ones(T)
    fade_len = int(T * 0.1)
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    weighted_signal = signal_2d * window.reshape(-1, 1)
    min_samples = int(T * 0.12) # Minimum duration constraint (~240ms)
    
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_samples).fit(weighted_signal)
    indices = algo.predict(n_bkps=n_bkps)
    
    return [time_ms[idx - 1] for idx in indices[:-1]]

def plot_academic_diagnostics(
    time_ms: np.ndarray,
    signal: np.ndarray,
    cps: list,
    title: str,
    ylabel: str,
    filename: str,
    task_type: str,
    line_color: str = "#1f77b4",
    output_dir: Path = None
):
    """
    Generates a unified, academic-style diagnostic plot with shaded cognitive states.
    The legend is placed completely outside the plot area to prevent data occlusion.
    """
    if output_dir is None:
        output_dir = get_project_paths()["PLOTS"]
        
    fig, ax = plt.subplots(figsize=(12, 5))

    # Signal smoothing for trend visualization
    smooth_w = max(len(signal) // 20, 5)
    smoothed = np.convolve(signal, np.ones(smooth_w) / smooth_w, mode="same")

    # Plot signals
    ax.plot(time_ms, smoothed, color=line_color, linewidth=2.5, label=f"Smoothed {ylabel}")
    ax.plot(time_ms, signal, color=line_color, alpha=0.3, linewidth=1, label="Raw Signal")

    # Plot Change Points (Vertical Lines)
    for cp in cps:
        ax.axvline(x=cp, color="#d62728", linestyle="--", linewidth=2, alpha=0.9)

    # Shaded Physiological Regions (Requires exactly 4 CPs to create 5 regions)
    if len(cps) == 4:
        b0, b1, b2, b3, b4, b5 = time_ms[0], cps[0], cps[1], cps[2], cps[3], time_ms[-1]

        # Dynamically define labels based on task_type (ERN vs CRN)
        task_label = "ERN" if task_type == "ERN" else "CRN"

        regions = [
            (b0, b1, "#cccccc", "Baseline"),
            (b1, b2, "#ffbb78", f"Pre-{task_label}\n(Prep)"),
            (b2, b3, "#ff9896", f"Task Core\n({task_label})"),
            (b3, b4, "#98df8a", f"Post-{task_label}\n(Processing)"),
            (b4, b5, "#aec7e8", "Recovery")
        ]

        for start, end, color, label in regions:
            ax.axvspan(start, end, facecolor=color, alpha=0.35)
            mid_pt = start + (end - start) / 2
            
            # Place centered text within each region
            ax.text(
                mid_pt,
                ax.get_ylim()[1] * 0.85,
                label,
                ha="center",
                va="top",
                fontsize=10,
                color="#222222",
                weight="bold"
            )

    # Formatting and Labels
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim([time_ms[0], time_ms[-1]])
    ax.grid(True, linestyle=":", alpha=0.6)

    # Mark the Action/Response at 0ms
    ax.axvline(x=0, color="black", linestyle="-", linewidth=2.5, label="Action (0ms)")

    # Legend Placement: Move completely outside the plot boundaries
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    # Save and clean up
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.png")
    
    # Ensure the external legend is not cut off during export using bbox_inches="tight"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {filepath}")
