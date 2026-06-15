import os
from pathlib import Path
import numpy as np
import common

class DMDTracker:
    def __init__(self, window_size, rank=5):
        self.window_size = window_size
        self.rank = rank

    def _compute_dmd(self, window):
        # Average over subjects (axis 2) -> (nodes, nodes, time)
        X = np.mean(window, axis=2)
        X_flat = X.reshape(X.shape[0] * X.shape[1], -1)

        X1 = X_flat[:, :-1]
        X2 = X_flat[:, 1:]

        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        r = min(self.rank, X1.shape[1])

        A_tilde = U[:, :r].T @ X2 @ Vh[:r, :].T / s[:r]
        evals = np.linalg.eigvals(A_tilde)
        return np.sort(np.abs(evals))

    def track(self, tensor):
        T = tensor.shape[-1]
        distances = np.zeros(T)
        prev_mode = None

        for t in range(self.window_size, T):
            window = tensor[..., t - self.window_size:t]
            curr_mode = self._compute_dmd(window)
            if prev_mode is not None:
                distances[t] = np.linalg.norm(curr_mode - prev_mode)
            prev_mode = curr_mode

        distances[:self.window_size] = distances[self.window_size]
        return distances

def main():
    paths = common.get_project_paths()
    os.makedirs(paths["DMD_OUT"], exist_ok=True)
    
    tensors = common.load_available_tensors()
    dmd_results = {}
    
    # window size for 1024Hz tensors
    win = 80

    for label, (tensor_data, time_ms) in tensors.items():
        print(f"\nRunning DMD Tracking | {label}")
        print("-" * 60)
        
        tracker = DMDTracker(window_size=win, rank=5)
        dist_signal = tracker.track(tensor_data)
        cps = common.extract_physiological_cps(dist_signal, time_ms)
        dmd_results[label] = cps
        
        print(f"  -> Detected CPs (ms): {[round(x, 1) for x in cps]}")
        
        # Save academic diagnostic plot
        common.plot_academic_diagnostics(
            time_ms=time_ms, 
            signal=dist_signal, 
            cps=cps,
            title=f"DMD: Eigenvalue Shift ({label})",
            ylabel="Eigenvalue Shift",
            filename=f"DMD_{label}_Plot",
            task_type=label.split("_")[-1],
            output_dir=paths["PLOTS"]
        )

    # Save outputs
    save_path = paths["DMD_OUT"] / "DMD_ChangePoints.npy"
    np.save(save_path, dmd_results)
    print(f"Aggregated DMD change points saved to {save_path}")

if __name__ == "__main__":
    main()
