"""
Configuration and hyperparameters for the vision-language model.
"""

from dataclasses import dataclass


@dataclass
class CFG:
    """
    Training and model configuration.

    Hyperparameters based on research.md decisions:
    - Image size: 512×512 baseline (768×768 upgrade option)
    - Model: ResNet18 + GRU + Cross-Attention
    - Loss: Smooth L1 (GIoU upgrade path)
    - Optimizer: AdamW with cosine scheduler
    """

    # Core settings
    IMG_SIZE: int = 512
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 8
    SEED: int = 42
    DIM: int = 256
    NUM_WORKERS: int = 2
    NO_PRETRAIN: bool = False  # True → disable ImageNet weights

    # Data paths
    TRAIN_JSON_DIR: str = "./data/train/report_json"
    TRAIN_JPG_DIR: str = "./data/train/report_jpg"
    VAL_JSON_DIR: str = "./data/val/report_json"
    VAL_JPG_DIR: str = "./data/val/report_jpg"
    TEST_JSON_DIR: str = "./data/test/query"
    TEST_JPG_DIR: str = "./data/test/images"

    # Output paths
    CKPT_PATH: str = "./outputs/ckpt/cross_attn_vlm.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"

    # Model architecture
    BACKBONE: str = "resnet18"  # resnet18, resnet50, efficientnet_b3
    TEXT_ENCODER: str = "gru"  # gru, clip
    FUSION_TYPE: str = "cross_attn"  # cross_attn, film
    FUSION_HEADS: int = 1  # For multi-head attention

    # Training options
    SCHEDULER: str = "cosine"  # cosine, step, plateau
    LOSS_TYPE: str = "smooth_l1"  # smooth_l1, giou, combined
    WEIGHT_DECAY: float = 1e-4

    # Augmentation
    COLOR_JITTER: bool = False  # Enable ColorJitter augmentation
    BRIGHTNESS: float = 0.2
    CONTRAST: float = 0.2
    SATURATION: float = 0.1
    ROTATION_DEGREES: float = 5.0

    # Misc
    PIN_MEMORY: bool = True
    SAVE_FREQ: int = 5  # Save checkpoint every N epochs
    EARLY_STOPPING_PATIENCE: int = 5
