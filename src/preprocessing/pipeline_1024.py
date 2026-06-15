import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import compute_current_source_density
from typing import List, Optional
from preprocessing.config_1024 import config

def preprocess_individual_subject(subject_id: str) -> mne.Epochs:
  """
  Preprocess EEG data keeping the original 1024Hz sampling rate:
  - Current Source Density (CSD) transformation applied.
  - No downsampling (SFREQ = 1024).
  """
  raw_path: Path = config.paths.DATA_RAW / subject_id / "eeg" / f"{subject_id}_task-ERN_eeg.set"
  if not raw_path.exists():
    raise FileNotFoundError(f"File not found: {raw_path}")
    
  raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
  
  # Rename & Montage
  rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
  raw.rename_channels(lambda x: rename_map.get(x, x))
  raw.set_montage(config.eeg.MONTAGE_NAME, on_missing='ignore')
  
  # Resample if and only if original is different from config.eeg.SFREQ
  if raw.info['sfreq'] != config.eeg.SFREQ:
    raw.resample(config.eeg.SFREQ)
  
  # Pick Channels
  available_eeg: List[str] = [ch for ch in config.eeg.CHANNELS if ch in raw.ch_names]
  raw_eeg_only = raw.copy().pick(available_eeg)
  
  # CSD Transformation
  raw_csd = compute_current_source_density(raw_eeg_only, verbose=False)
  
  # Epoching
  events, event_id = mne.events_from_annotations(raw_csd, verbose=False)
  correct_codes: List[str] = [l for l in event_id.keys() if len(l) == 3 and l[0] == l[2]]
  incorrect_codes: List[str] = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
  
  resp_id = {label: event_id[label] for label in correct_codes + incorrect_codes}
  epochs = mne.Epochs(
    raw_csd, events, event_id=resp_id, 
    tmin=config.eeg.TMIN, tmax=config.eeg.TMAX, 
    baseline=None, preload=True, 
    detrend=None, verbose=False
  )
  
  mapping = {l: f"Correct/{l}" for l in correct_codes}
  mapping.update({l: f"Incorrect/{l}" for l in incorrect_codes})
  epochs.event_id = {mapping[k]: v for k, v in epochs.event_id.items()}
  
  save_file: Path = config.paths.EPOCHS_DIR / f"{subject_id}_master-epo.fif"
  epochs.save(save_file, overwrite=True, verbose=False)
  return epochs

if __name__ == "__main__":
  sub_ids: List[str] = sorted([d.name for d in config.paths.DATA_RAW.glob("sub-*") if d.is_dir()])
  
  report_data: List[dict] = []
  print(f"Running preprocessing (1024Hz) for {len(sub_ids)} subjects...")
  
  for i, sub in enumerate(sub_ids):
    try:
      ep = preprocess_individual_subject(sub)
      report_data.append({
        'sub': sub, 
        'status': 'SUCCESS', 
        'correct': len(ep['Correct']), 
        'incorrect': len(ep['Incorrect'])
      })
      print(f"  {sub}: Preprocessing success. Correct: {len(ep['Correct'])}, Incorrect: {len(ep['Incorrect'])}")
    except Exception as e:
      report_data.append({'sub': sub, 'status': f'FAILED: {e}', 'correct': 0, 'incorrect': 0})
      print(f"  {sub}: FAILED | {e}")
      
  pd.DataFrame(report_data).to_csv(config.report_etl, index=False)
  print(f"Completed! Report saved at: {config.report_etl}")
