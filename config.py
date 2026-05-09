import os

class Config:
    DATA_PATH = 'path/to/ATRIA/Training_Set'
    SAVE_PATH = './checkpoints'
    HISTORY_DIR = './history'
    PLOTS_DIR = './plots'

    INPUT_SIZE = 128
    HIDDEN_SIZE = 256
    BOTTLENECK_DIM = 128
    GRANULE_EXPANSION = 20
    TAU_VALUES = [1, 2, 3, 4]
    CORTICAL_ALPHA = 0.1

    FEEDBACK_TYPE = 'cerebellar'   # 'readout', 'cerebellar'

    FREEZE_RECURRENT = True
    FREEZE_INPUT = False
    FREEZE_CEREBELLAR_INPUT = False
    FREEZE_ENCODER = False

    CEREBELLUM_LR_MULT = 2.7
    CEREB_LOSS_MAX_WEIGHT = 0.4
    CEREB_LOSS_WARMUP_EPOCHS = 3
    CEREB_LOSS_RAMPUP_EPOCHS = 7

    NORMALIZE_FEATURES = True
    CLIP_FEEDBACK = False
    FEEDBACK_ACT = 'tanh'
    GRANULE_DROPOUT = 0.2

    USE_AUGMENTATION = True        # False for clean (non-noisy) data
    SLICE_STEP = 7                 # 1 for full slice sequence (no subsampling)
    AUGMENTATION_PROB = 0.7
    NOISE_STD = 0.2
    SHIFT_MAX = 12

    MAX_SLICES_PER_VOLUME = 96

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.2
    NUM_EPOCHS = 50
    NUM_WORKERS = 2
    RANDOM_SEED = 42

    FOCAL_GAMMA = 2.0
    TVERSKY_ALPHA = 0.7
    TVERSKY_BETA = 0.3

    @staticmethod
    def create_dirs():
        os.makedirs(Config.SAVE_PATH, exist_ok=True)
        os.makedirs(Config.HISTORY_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    @staticmethod
    def set_clean_mode():
        Config.USE_AUGMENTATION = False
        Config.SLICE_STEP = 1

    @staticmethod
    def set_corrupted_mode():
        Config.USE_AUGMENTATION = True
        Config.SLICE_STEP = 7

cfg = Config()