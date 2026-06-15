import mne
import numpy as np
import torch
from pathlib import Path
from scipy.io import savemat
from typing import List, Optional, Tuple
from preprocessing.config_1024 import config
from algorithms.time_frequency import RIDRihaczek


def compute_group_connectivity() -> None:
  """
  Step 3: Compute PLV tensors from already sampled and filtered epochs at 1024Hz.
  """
  sub_files = sorted(list(config.paths.REFINED_DIR.glob("*_theta_balanced-epo.fif")))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if not sub_files:
    print("Error: No sampled epoch files found. Please run sampling_1024.py first.")
    return
    
  processed_subs: List[str] = []
  all_corr: List[np.ndarray] = []
  all_inc: List[np.ndarray] = []
  
  sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
  n_channels = len(sample_epochs.ch_names)
  n_points = sample_epochs.get_data().shape[2]

  print(f"Computing PLV Tensors (Step 3) for {len(sub_files)} subjects on {device}...")

  batch_size_trials = 32 # Keep batch size small to avoid CUDA OOM

  for i, f in enumerate(sub_files):
    subject_id = f.name.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    
    processed_subs.append(subject_id)
    
    sub_plv_cor = np.zeros((n_channels, n_channels, n_points), dtype=np.float32)
    sub_plv_inc = np.zeros((n_channels, n_channels, n_points), dtype=np.float32)

    tfd_engine = RIDRihaczek(n_points)
    
    # Precompute theta mask
    freq_axis = torch.fft.fftfreq(n_points, d=1/config.eeg.SFREQ, device=device)
    theta_mask = (freq_axis >= config.proc.THETA_BAND[0]) & (freq_axis <= config.proc.THETA_BAND[1])
    theta_freq_indices = torch.where(theta_mask)[0]

    for condition, storage in [('Correct', sub_plv_cor), ('Incorrect', sub_plv_inc)]:
      data = torch.tensor(epochs[condition].get_data(), dtype=torch.float32, device=device)
      n_trials, n_channels, n_points = data.shape
      if n_trials == 0: continue
      
      # We will store the normalized complex values (phase) for the theta band
      # Shape: (n_trials, n_channels, n_points, n_theta_freqs)
      theta_phases = torch.zeros((n_trials, n_channels, n_points, len(theta_freq_indices)), 
                                 dtype=torch.complex64, device=device)
      
      for ch in range(n_channels):
          # Compute TFD in batches of trials to avoid GPU OOM for 1024Hz data
          for start_idx in range(0, n_trials, batch_size_trials):
              end_idx = min(start_idx + batch_size_trials, n_trials)
              trial_data = data[start_idx:end_idx, ch, :]
              tfr = tfd_engine.compute_tfd(trial_data) # (batch, time, freq)
              # Extract theta frequencies and normalize
              theta_samples = tfr[:, :, theta_mask] # (batch, time, n_theta)
              theta_phases[start_idx:end_idx, ch, :, :] = theta_samples / (torch.abs(theta_samples) + 1e-12)
    
      # PLV = |mean_trials( exp(j * (phi1 - phi2)) )|
      # We average PLV over the theta frequencies as per standard TFD-PLV
      for ch1 in range(n_channels):
          for ch2 in range(ch1 + 1, n_channels):
              # Complex product across trials for each (time, freq)
              # Result shape: (n_points, n_theta)
              sync = torch.abs(torch.mean(theta_phases[:, ch1, :, :] * torch.conj(theta_phases[:, ch2, :, :]), dim=0))
              # Average over the theta frequency bins
              plv_time_series = torch.mean(sync, dim=1)
              storage[ch1, ch2, :] = plv_time_series.cpu().numpy()
              storage[ch2, ch1, :] = plv_time_series.cpu().numpy()
     
    all_corr.append(sub_plv_cor)
    all_inc.append(sub_plv_inc)
    print(f"  {subject_id}: Completed PLV calculation.")

  tensor_correct = np.stack(all_corr, axis=0)
  tensor_incorrect = np.stack(all_inc, axis=0)
  
  # Save results
  np.save(config.tensor_correct_file, tensor_correct)
  np.save(config.tensor_incorrect_file, tensor_incorrect)
  
  # Make sure MATLAB dir exists
  config.paths.MATLAB_DIR.mkdir(parents=True, exist_ok=True)
  savemat(config.paths.MATLAB_DIR / "connectivity_balanced_4d.mat", {
    'tensor_correct': tensor_correct,
    'tensor_incorrect': tensor_incorrect,
    'subjects': processed_subs,
    'channels': sample_epochs.ch_names
  })
  
  print(f"Completed! Processed {len(processed_subs)} subjects. Tensors saved to {config.paths.TENSOR_DIR}")

if __name__ == "__main__":
  compute_group_connectivity()
