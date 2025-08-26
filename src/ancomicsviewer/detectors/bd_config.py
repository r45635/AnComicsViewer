# bd_config.py
class BDConfig:
    """Configuration BD optimisée après tests."""
    CONF_BASE = 0.05
    CONF_MIN = 0.01
    IOU_NMS = 0.50
    MAX_DET   = 300
    TTA       = False

    # adaptive filter
    TARGET_MIN = 3
    TARGET_MAX = 24

    # priors
    MIN_AREA_RATIO = 0.02
    ASPECT_MIN = 0.2
    ASPECT_MAX = 5.0

    # WBF / merges
    WBF_IOU = 0.65
    MERGE_OVERLAP = 0.85
    MERGE_GAP     = 0.02

    # split
    SPLIT_LARGE_RATIO = 0.35
    SPLIT_VALLEY_LEN  = 0.80
