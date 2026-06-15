import os
from pathlib import Path
import numpy as np
import common

class HORLSLTracker:
    def __init__(self, alpha, sigma_min=0.11):
        self.alpha = alpha
        self.sigma_min = sigma_min
        self.P = [None, None, None]

    def _unfold(self, tensor, mode):
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def fit_transform(self, tensor_data, train_time):
        N1, N2, S, T = tensor_data.shape
        low_rank = np.zeros_like(tensor_data)
        energy = np.zeros(T)

        # Initialize subspaces
        init_data = tensor_data[..., :train_time]
        for mode in range(3):
            U, S_vals, _ = np.linalg.svd(self._unfold(init_data, mode), full_matrices=False)
            r = np.sum(S_vals > self.sigma_min * S_vals[0])
            self.P[mode] = U[:, :max(r, 1)]

        low_rank[..., :train_time] = init_data

        # Recursive tracking
        for t in range(train_time, T):
            H_t = tensor_data[..., t]
            sparse_t = H_t.copy()

            for mode in range(3):
                P_mat = self.P[mode]
                I_minus_PPT = np.eye(P_mat.shape[0]) - P_mat @ P_mat.T
                unfolded_sparse = self._unfold(sparse_t, mode)
                projected = I_minus_PPT @ unfolded_sparse

                if mode == 0:
                    sparse_t = projected.reshape((N1, N2, S))
                elif mode == 1:
                    sparse_t = projected.reshape((N2, N1, S)).transpose(1, 0, 2)
                else:
                    sparse_t = projected.reshape((S, N1, N2)).transpose(1, 2, 0)

            sparse_t = np.sign(sparse_t) * np.maximum(np.abs(sparse_t) - 0.01, 0)
            energy[t] = np.sqrt(np.sum(sparse_t**2))

            # Low-rank component (unprojected residual)
            L_noisy_t = H_t - sparse_t
            low_rank[..., t] = L_noisy_t

            # Update subspaces
            if t % self.alpha == 0:
                recent_data = low_rank[..., t - self.alpha:t]
                for mode in range(3):
                    U, S_vals, _ = np.linalg.svd(self._unfold(recent_data, mode), full_matrices=False)
                    r = np.sum(S_vals > self.sigma_min * S_vals[0])
                    self.P[mode] = U[:, :max(r, 1)]

        return low_rank, energy

def evaluate_cps(cps):
    """Scoring metric: Maximizes the minimum distance between CPs."""
    if len(cps) < 2:
        return 0
    return np.min(np.diff(cps))

def main():
    paths = common.get_project_paths()
    os.makedirs(paths["HORLSL_OUT"], exist_ok=True)
    
    tensors = common.load_available_tensors()
    horlsl_all_cps = {}
    
    # 1024Hz parameters
    alphas_to_test = [32, 64, 128]
    sigmas_to_test = [0.05, 0.11, 0.20]
    train_time = 80
    base_alpha = alphas_to_test[1]

    for label, (tensor_data, time_ms) in tensors.items():
        print(f"\nOptimization & Grid Search | HO-RLSL | {label}")
        print("-" * 60)
        
        best_horlsl = {'score': -1, 'params': None, 'cps': None, 'lr': None, 'energy': None}
        
        # Sweep alpha
        for a in alphas_to_test:
            tracker = HORLSLTracker(alpha=a, sigma_min=sigmas_to_test[1])
            lr, energy = tracker.fit_transform(tensor_data, train_time=train_time)
            cps = common.extract_physiological_cps(energy, time_ms)
            score = evaluate_cps(cps)
            
            print(f"  Alpha={a:<3d}, Sigma={sigmas_to_test[1]:.2f} | CPs: {[round(x, 1) for x in cps]} | Spread Score: {score:.1f}")
            
            if score > best_horlsl['score']:
                best_horlsl = {'score': score, 'params': {'alpha': a, 'sigma': sigmas_to_test[1]}, 'cps': cps, 'lr': lr, 'energy': energy}

        # Sweep sigma
        for sig in sigmas_to_test:
            if sig == sigmas_to_test[1]: 
                continue
            tracker = HORLSLTracker(alpha=base_alpha, sigma_min=sig)
            lr, energy = tracker.fit_transform(tensor_data, train_time=train_time)
            cps = common.extract_physiological_cps(energy, time_ms)
            score = evaluate_cps(cps)
            
            print(f"  Alpha={base_alpha:<3d}, Sigma={sig:.2f} | CPs: {[round(x, 1) for x in cps]} | Spread Score: {score:.1f}")
            
            if score > best_horlsl['score']:
                best_horlsl = {'score': score, 'params': {'alpha': base_alpha, 'sigma': sig}, 'cps': cps, 'lr': lr, 'energy': energy}

        opt_a = best_horlsl['params']['alpha']
        opt_s = best_horlsl['params']['sigma']
        print(f">> OPTIMAL CONFIG: Alpha={opt_a}, Sigma={opt_s} -> CPs: {[round(x, 1) for x in best_horlsl['cps']]}")
        
        # Save change points
        horlsl_all_cps[label] = best_horlsl['cps']
        
        # Save optimal LowRank matrix
        lr_save_path = paths["HORLSL_OUT"] / f"HORLSL_{label}_LowRank.npy"
        np.save(lr_save_path, best_horlsl['lr'])
        print(f"Exported optimal LowRank matrix to {lr_save_path}")
        
        # Save academic diagnostic plot
        common.plot_academic_diagnostics(
            time_ms=time_ms,
            signal=best_horlsl['energy'],
            cps=best_horlsl['cps'],
            title=f"HO-RLSL: Sparse Energy ({label})",
            ylabel="Sparse Energy",
            filename=f"HORLSL_{label}_Optimal_Plot",
            task_type=label.split("_")[-1],
            line_color="#d62728", # Red theme for HO-RLSL
            output_dir=paths["PLOTS"]
        )

    # Save aggregated change points dictionary
    cp_save_path = paths["HORLSL_OUT"] / "HORLSL_ChangePoints.npy"
    np.save(cp_save_path, horlsl_all_cps, allow_pickle=True)
    print(f"Aggregated HO-RLSL change points saved to {cp_save_path}")

if __name__ == "__main__":
    main()
