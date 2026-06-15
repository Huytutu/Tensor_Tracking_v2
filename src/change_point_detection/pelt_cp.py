import os
from pathlib import Path
import numpy as np
import ruptures as rpt
import common

def pelt_search(tensor_data: np.ndarray, time_ms: np.ndarray, label: str, penalties: list):
    """
    Sweeps penalty values dynamically, identifies the optimal configuration,
    and returns the energy signal and optimal CPs.
    """
    print(f"\nOptimization & Grid Search | PELT | {label}")
    print("-" * 60)
    
    # Extract global energy feature (Frobenius Norm)
    mean_tensor = np.mean(tensor_data, axis=2) 
    global_energy = np.sqrt(np.sum(mean_tensor**2, axis=(0, 1))) 
    signal = global_energy.reshape(-1, 1)
    
    T_total = len(global_energy)
    min_size = int(T_total * 0.08) 
    
    best_cps = []
    best_pen = None
    
    # Grid Search
    for pen in penalties:
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
        result_indices = algo.predict(pen=pen)
        
        cps_idx = result_indices[:-1] # Remove the last index (end of data)
        cps_ms = [time_ms[idx - 1] for idx in cps_idx]
        
        print(f"  [Penalty = {pen:5.1f}] -> Found {len(cps_ms)} points: {[round(x, 1) for x in cps_ms]} ms")
        
        # Automatically save the first configuration that yields 3 or 4 points
        if len(cps_ms) in [3, 4] and best_pen is None:
            best_cps = cps_ms
            best_pen = pen
            
    if best_pen is None:
        # Fallback to the last penalty if none yielded 3 or 4 points
        best_cps = [time_ms[idx - 1] for idx in cps_idx]
        best_pen = penalties[-1]
        
    print(f">> OPTIMAL CONFIG: Penalty = {best_pen} -> CPs: {[round(x, 1) for x in best_cps]}")
    return global_energy, best_cps

def main():
    paths = common.get_project_paths()
    os.makedirs(paths["PELT_OUT"], exist_ok=True)
    
    tensors = common.load_available_tensors()
    pelt_all_cps = {}
    
    # 1024Hz penalties
    penalties = [5.0, 10.0, 20.0, 50.0, 100.0]

    for label, (tensor_data, time_ms) in tensors.items():
        energy_signal, opt_cps = pelt_search(tensor_data, time_ms, label, penalties)
        
        # Store in dictionary
        pelt_all_cps[label] = opt_cps
        
        # Generate diagnostic plot in green theme
        common.plot_academic_diagnostics(
            time_ms=time_ms,
            signal=energy_signal,
            cps=opt_cps,
            title=f"PELT Optimal: Global Energy ({label})",
            ylabel="Global Energy (Frobenius Norm)",
            filename=f"PELT_{label}_Optimal_Plot",
            task_type=label.split("_")[-1],
            line_color="#2ca02c", # Green theme for PELT
            output_dir=paths["PLOTS"]
        )

    # Save aggregated dictionary
    save_path = paths["PELT_OUT"] / "PELT_ChangePoints.npy"
    np.save(save_path, pelt_all_cps, allow_pickle=True)
    print(f"Aggregated PELT change points saved to {save_path}")

if __name__ == "__main__":
    main()
