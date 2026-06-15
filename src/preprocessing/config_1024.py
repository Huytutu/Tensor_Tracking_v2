from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass(frozen=True)
class PathConfig:
  """Directory structure configuration for 1024Hz experiment."""
  BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
  DATA_RAW: Path = BASE_PATH / "data" / "ERN_Raw_Data_BIDS-Compatible"
  DATA_PROCESSED: Path = BASE_PATH / "outputs" / "processed_1024"
  
  # Sub-directories
  EPOCHS_DIR: Path = DATA_PROCESSED / "01_initial_epochs"
  REFINED_DIR: Path = DATA_PROCESSED / "02_balanced_theta_epochs"
  TENSOR_DIR: Path = DATA_PROCESSED / "03_connectivity_tensors"
  TRACKING_DIR: Path = DATA_PROCESSED / "04_tracking_results"
  MATLAB_DIR: Path = DATA_PROCESSED / "05_matlab_results"
  OUTPUTS_DIR: Path = BASE_PATH / "outputs" / "eda_1024"

  def __post_init__(self):
    """Create directories if they do not exist."""
    for d in [self.EPOCHS_DIR, self.REFINED_DIR, self.TENSOR_DIR, self.TRACKING_DIR, self.MATLAB_DIR, self.OUTPUTS_DIR]:
      d.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class EEGConfig:
  """EEG Signal processing parameters (Ozdemir 2017) at 1024Hz."""
  SFREQ: int = 1024 # Keep 1024 Hz, no downsampling
  TMIN: float = -1.0
  TMAX: float = 1.0
  BASELINE: Tuple[float, float] = (-0.6, -0.4)
  
  # 30 EEG Channels for high spatial resolution
  CHANNELS: List[str] = field(default_factory=lambda: [
    'Fp1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz',
    'Fp2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2'
  ])
  
  EOG_CHANNELS: List[str] = field(default_factory=lambda: ['HEOG_left', 'HEOG_right', 'VEOG_lower'])
  MONTAGE_NAME: str = "standard_1005"

@dataclass(frozen=True)
class ProcessConfig:
  """Processing and algorithm thresholds."""
  FILTER_LOW: float = 0.1
  FILTER_HIGH: float = 30.0
  THETA_BAND: Tuple[float, float] = (4.0, 8.0) # Hz
  
  # ICA
  ICA_COMPONENTS: int = 10
  ICA_METHOD: str = 'infomax'
  
  # Minimum trials for inclusion
  MIN_INCORRECT_TRIALS: int = 1

@dataclass(frozen=True)
class AlgorithmConfig:
  """Denoising, change point tracking, and clustering parameters."""
  # Low rank extraction
  HO_RLSL_TRAIN_LENGTH: int = 80
  HO_RLSL_ALPHA: int = 64
  HO_RLSL_SIGMA_MIN: float = 0.11
  HO_RLSL_SPARSE_SOLVER: str = "gtcs_s_omp"
  HOSVD_RANKS: Tuple[int, int, int] = (10, 10, 10)
  
  # Change Point Detection
  N_CHANGE_POINTS: int = 2
  MIN_INTERVAL_FRAMES: int = 10
  
  # FCCA Consensus Clustering
  FCCA_K_RANGE: Tuple[int, ...] = (2, 3, 4, 5)
  FCCA_MIN_COMMUNITY_SIZE: int = 2

@dataclass(frozen=True)
class GlobalConfig:
  """Consolidated configuration for 1024Hz."""
  paths: PathConfig = field(default_factory=PathConfig)
  eeg: EEGConfig = field(default_factory=EEGConfig)
  proc: ProcessConfig = field(default_factory=ProcessConfig)
  algo: AlgorithmConfig = field(default_factory=AlgorithmConfig)
  
  # Output Files
  @property
  def tensor_correct_file(self) -> Path:
    return self.paths.TENSOR_DIR / "tensor_correct_4d.npy"
    
  @property
  def tensor_incorrect_file(self) -> Path:
    return self.paths.TENSOR_DIR / "tensor_incorrect_4d.npy"
    
  @property
  def report_etl(self) -> Path:
    return self.paths.DATA_PROCESSED / "master_preprocessing_report.csv"

# Singleton instance
config = GlobalConfig()
