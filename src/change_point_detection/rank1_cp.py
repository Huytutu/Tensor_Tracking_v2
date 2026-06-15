import os
from pathlib import Path
import numpy as np
import common

class Rank1CPTracker:
    def __init__(self, window_size: int):
        self.window_size = window_size

    def _estimate_spatial_signature(self, window: np.ndarray) -> np.ndarray:
        """Estimates the core spatial signature using tensor summation."""
        u = np.sum(window, axis=(1, 2, 3))
        return u / (np.linalg.norm(u) + 1e-8)

    def track(self, tensor: np.ndarray) -> np.ndarray:
        """Tracks structural changes using subspace distance."""
        n_time = tensor.shape[-1]
        distances = np.zeros(n_time)
        prev_u = None

        for t in range(self.window_size, n_time):
            window = tensor[..., t - self.window_size:t]
            curr_u = self._estimate_spatial_signature(window)
            
            if prev_u is not None:
                # Calculate subspace shift avoiding sign ambiguity
                overlap = np.abs(np.dot(curr_u, prev_u))
                distances[t] = 1.0 - overlap
                
            prev_u = curr_u
            
        distances[:self.window_size] = distances[self.window_size]
        return distances

def main():
    paths = common.get_project_paths()
    os.makedirs(paths["CP_OUT"], exist_ok=True)
    
    tensors = common.load_available_tensors()
    cp_results = {}
    
    # window size for 1024Hz tensors
    win = 80

    for label, (tensor_data, time_ms) in tensors.items():
        print(f"\nRunning Rank-1 CP Tracking | {label}")
        print("-" * 60)
        
        tracker = Rank1CPTracker(window_size=win)
        dist_signal = tracker.track(tensor_data)
        cps = common.extract_physiological_cps(dist_signal, time_ms)
        cp_results[label] = cps
        
        print(f"  -> Detected CPs (ms): {[round(x, 1) for x in cps]}")
        
        # Save academic diagnostic plot
        common.plot_academic_diagnostics(
            time_ms=time_ms, 
            signal=dist_signal, 
            cps=cps,
            title=f"Rank-1 CP Tracking: Subspace Shift ({label})",
            ylabel="Subspace Distance (1 - |dot|)",
            filename=f"CP_Tracking_{label}_Optimal",
            task_type=label.split("_")[-1],
            output_dir=paths["PLOTS"]
        )

    # Save outputs
    save_path = paths["CP_OUT"] / "CP_ChangePoints.npy"
    np.save(save_path, cp_results)
    print(f"Aggregated Rank-1 change points saved to {save_path}")

if __name__ == "__main__":
    main()
