import os
from pathlib import Path
import numpy as np
import common

class HoSVDAlgorithm:
    """
    Sliding-window HOSVD-based change point detection.

    Parameters
    ----------
    window_size : int
        Temporal window length.

    rank : int, default=5
        Subspace rank used for Grassmann distance computation.
    """

    def __init__(self, window_size, rank=5):
        self.window_size = window_size
        self.rank = rank

    def _get_subspace(self, tensor_window):
        """Extract dominant subspace from a tensor window."""
        mat = np.moveaxis(
            tensor_window,
            0,
            0
        ).reshape(
            tensor_window.shape[0],
            -1
        )
        U, _, _ = np.linalg.svd(
            mat,
            full_matrices=False
        )
        return U[:, :self.rank]

    def fit_transform(self, tensor_data):
        """
        Compute change-point signal and low-rank reconstruction.

        Parameters
        ----------
        tensor_data : ndarray
            Tensor with shape (Nodes, Nodes, Subjects, Time)

        Returns
        -------
        distances : ndarray
            Grassmann distance trajectory.

        low_rank : ndarray
            Low-rank reconstruction.

        residual : ndarray
            Residual component.
        """
        T = tensor_data.shape[-1]
        distances = np.zeros(T)
        low_rank = np.zeros_like(tensor_data)
        residual = np.zeros_like(tensor_data)
        prev_U = None
        step = max(self.window_size // 2, 1)

        for t in range(self.window_size, T, step):
            window = tensor_data[..., t - self.window_size:t]
            curr_U = self._get_subspace(window)

            if prev_U is not None:
                overlap = np.trace(
                    (curr_U.T @ prev_U)
                    @
                    (prev_U.T @ curr_U)
                )
                dist = 1.0 - (overlap / self.rank)
                distances[t - step // 2] = dist

            prev_U = curr_U
            center_idx = t - step // 2
            H_t = tensor_data[..., center_idx]
            P_mat = curr_U @ curr_U.T
            unfolded_H = np.moveaxis(
                H_t,
                0,
                0
            ).reshape(
                H_t.shape[0],
                -1
            )
            proj_L = P_mat @ unfolded_H
            low_rank_t = proj_L.reshape(H_t.shape)
            low_rank[..., center_idx] = low_rank_t
            residual[..., center_idx] = H_t - low_rank_t

        return distances, low_rank, residual

def evaluate_cps(cps):
    """Scoring metric: Maximizes the minimum distance between CPs."""
    if len(cps) < 2:
        return 0
    return np.min(np.diff(cps))

def main():
    paths = common.get_project_paths()
    os.makedirs(paths["HOSVD_OUT"], exist_ok=True)
    
    # Load 1024Hz tensors (default)
    tensors = common.load_available_tensors()
    
    hosvd_all_cps = {}
    
    # We will test windows [40, 80, 160] for 1024Hz tensors
    windows_to_test = [40, 80, 160]
    
    for label, (tensor_data, time_ms) in tensors.items():
        print(f"\nOptimization & Grid Search | HOSVD | {label}")
        print("-" * 60)
        
        best_hosvd = {'score': -1, 'params': None, 'cps': None, 'lr': None, 'dist': None}
        
        for w in windows_to_test:
            algo = HoSVDAlgorithm(window_size=w, rank=5)
            dist, lr, _ = algo.fit_transform(tensor_data)
            cps = common.extract_physiological_cps(dist, time_ms)
            score = evaluate_cps(cps)
            
            print(f"  Window={w:<3d} | CPs: {[round(x, 1) for x in cps]} | Spread Score: {score:.1f}")
            
            if score > best_hosvd['score']:
                best_hosvd = {'score': score, 'params': {'window': w}, 'cps': cps, 'lr': lr, 'dist': dist}
                
        opt_w = best_hosvd['params']['window']
        print(f">> OPTIMAL CONFIG: Window={opt_w} -> CPs: {[round(x, 1) for x in best_hosvd['cps']]}")
        
        # Save change points
        hosvd_all_cps[label] = best_hosvd['cps']
        
        # Save the bulky LowRank matrix to disk
        lr_save_path = paths["HOSVD_OUT"] / f"HOSVD_{label}_LowRank.npy"
        np.save(lr_save_path, best_hosvd['lr'])
        print(f"Exported optimal LowRank matrix to {lr_save_path}")
        
        # Save academic diagnostic plot
        common.plot_academic_diagnostics(
            time_ms=time_ms,
            signal=best_hosvd['dist'],
            cps=best_hosvd['cps'],
            title=f"HOSVD: Grassmann Distance ({label})",
            ylabel="Grassmann Distance",
            filename=f"HOSVD_{label}_Optimal_Plot",
            task_type=label.split("_")[-1],
            line_color="#1f77b4",
            output_dir=paths["PLOTS"]
        )

    # Save aggregated change points dictionary
    cp_save_path = paths["HOSVD_OUT"] / "HOSVD_ChangePoints.npy"
    np.save(cp_save_path, hosvd_all_cps, allow_pickle=True)
    print(f"Aggregated HOSVD change points saved to {cp_save_path}")

if __name__ == "__main__":
    main()
