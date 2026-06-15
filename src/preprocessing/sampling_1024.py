import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from preprocessing.config_1024 import config

def match_temporal_trials(epochs: mne.Epochs) -> Optional[mne.Epochs]:
  """
  Balance Correct and Incorrect trials using temporal matching to avoid biological drift bias.
  """
  events = epochs.events
  event_id = epochs.event_id
  
  # Get indices for Correct and Incorrect trials
  correct_idx = np.where(np.isin(events[:, 2], [event_id[k] for k in event_id if k.startswith('Correct')]))[0]
  incorrect_idx = np.where(np.isin(events[:, 2], [event_id[k] for k in event_id if k.startswith('Incorrect')]))[0]
  
  if len(incorrect_idx) == 0:
    return None

  matched_correct_idx = []
  for inc_idx in incorrect_idx:
    inc_time = events[inc_idx, 0]
    # Find the nearest Correct trial (before or after)
    distances = np.abs(events[correct_idx, 0] - inc_time)
    nearest_cor_idx = correct_idx[np.argmin(distances)]
    matched_correct_idx.append(nearest_cor_idx)
    
    # Remove used correct trial to ensure 1:1 unique matching
    correct_idx = correct_idx[correct_idx != nearest_cor_idx]
    if len(correct_idx) == 0:
      break
      
  # Combine and sort by time
  final_indices = np.concatenate([incorrect_idx[:len(matched_correct_idx)], matched_correct_idx])
  final_indices.sort()
  
  return epochs[final_indices]

def run_sampling_pipeline() -> None:
  """
  Step 2: Filter in Theta band and apply Temporal Matching subsampling.
  """
  sub_files: List[Path] = sorted(list(config.paths.EPOCHS_DIR.glob("*_master-epo.fif")))
  print(f"Running Sampling (1024Hz + Temporal Matching) for {len(sub_files)} subjects...")
  
  valid_count = 0
  for f in sub_files:
    subject_id: str = f.stem.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    
    # Skip explicit filtering (Paper uses RID-Rihaczek on broadband signal)
    epochs_theta = epochs.copy()
    
    # Temporal Matching Subsampling
    balanced_epochs = match_temporal_trials(epochs_theta)
    
    if balanced_epochs is not None:
      save_file: Path = config.paths.REFINED_DIR / f"{subject_id}_theta_balanced-epo.fif"
      balanced_epochs.save(save_file, overwrite=True, verbose=False)
      valid_count += 1
      print(f"  {subject_id}: Successfully balanced trials ({len(balanced_epochs)} trials total).")
    else:
      print(f"  {subject_id}: Skipped, no incorrect trials found.")

  print(f"Sampling complete. Total subjects with matched trials: {valid_count}")

if __name__ == "__main__":
  run_sampling_pipeline()
