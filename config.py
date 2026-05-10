import os

class Config:

    # PATHS AND DIRECTORIES
    # Replace these placeholders with your actual local paths
    
    DATA_PATH = './path/to/ATRIA/Training_Set'
    SAVE_PATH = './checkpoints'
    HISTORY_DIR = './history'
    PLOTS_DIR = './plots'


    # ARCHITECTURE PARAMETERS (Section 2.3)

    INPUT_SIZE = 128
    HIDDEN_SIZE = 256
    BOTTLENECK_DIM = 128
    GRANULE_EXPANSION = 20          # Granule cell expansion factor (256 * 20 = 5120)
    TAU_VALUES = [1, 2, 3, 4]       # Temporal prediction horizons
    CORTICAL_ALPHA = 0.1            # Leaky RNN decay parameter
    NORMALIZE_FEATURES = True       # Apply LayerNorm to bottleneck features
    FEEDBACK_ACT = 'tanh'           # Activation function for cerebellar correction
    FEEDBACK_TYPE = 'cerebellar'    # Options: 'readout', 'cerebellar'

    # Weight plasticity configuration
    FREEZE_RECURRENT = True         # Freeze W_hh (fixed cortical dynamics)
    FREEZE_INPUT = False            # Trainable W_ih
    FREEZE_CEREBELLAR_INPUT = False # Trainable W_Ch
    FREEZE_ENCODER = False


    # CEREBELLAR TRAINING SCHEDULE (Section 2.4)

    CEREBELLUM_LR_MULT = 2.7
    CEREB_LOSS_MAX_WEIGHT = 0.4
    CEREB_LOSS_WARMUP_EPOCHS = 3
    CEREB_LOSS_RAMPUP_EPOCHS = 7
    GRANULE_DROPOUT = 0.2

    
    # DATA LOADING AND AUGMENTATIONS

    MAX_SLICES_PER_VOLUME = 96
    SLICE_STEP = 7                  # Slice subsampling factor (1 for full resolution)
    AUGMENTATION_PROB = 0.7         # Probability of applying train-time augmentations
    NOISE_STD = 0.2                 # Gaussian noise standard deviation
    SHIFT_MAX = 12                  # Maximum random spatial shift in pixels


    # TRAINING HYPERPARAMETERS

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.2
    NUM_EPOCHS = 50
    NUM_WORKERS = 2
    RANDOM_SEED = 42                # Default seed, explicitly set in main.py

    # Loss function coefficients
    FOCAL_GAMMA = 2.0
    TVERSKY_ALPHA = 0.7
    TVERSKY_BETA = 0.3

    # DataLoader optimization
    PIN_MEMORY = True

    @staticmethod
    def create_dirs():
        """Create output directories if they do not exist."""
        os.makedirs(Config.SAVE_PATH, exist_ok=True)
        os.makedirs(Config.HISTORY_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    @staticmethod
    def set_clean_mode():
        """Configure parameters for clean data experiments (Table 1)."""
        Config.SLICE_STEP = 1
        Config.AUGMENTATION_PROB = 0.0
        Config.NOISE_STD = 0.0
        Config.SHIFT_MAX = 0

    @staticmethod
    def set_corrupted_mode():
        """Configure parameters for degraded data experiments (Table 2)."""
        Config.SLICE_STEP = 7
        Config.AUGMENTATION_PROB = 0.7
        Config.NOISE_STD = 0.2
        Config.SHIFT_MAX = 12

cfg = Config()
