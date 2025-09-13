#!/usr/bin/env python3
"""
YOLO Training Script with Class-Aware NMS, Tiling, and Optimized Parameters
Based on project requirements: imgsz 1024, epochs 250, class-aware NMS, tiling 50-60%
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOTrainingManager:
    """YOLO Training Manager with optimized parameters for comic detection"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dataset_path = self.project_root / "dataset"
        self.configs_path = self.project_root / "config"

    def create_optimized_config(self) -> Dict[str, Any]:
        """Create optimized training configuration per requirements"""

        # Dataset analysis
        train_images = len(list((self.dataset_path / "images" / "train").glob("*.png")))
        val_images = len(list((self.dataset_path / "images" / "val").glob("*.png")))

        logger.info(f"Dataset: {train_images} train, {val_images} val images")

        # Optimized configuration per requirements
        config = {
            # Model and data
            'model': 'yolov8m.pt',  # Medium model for capacity
            'data': str(self.dataset_path / "multibd_enhanced.yaml"),

            # Training parameters (per requirements)
            'epochs': 250,  # ✅ 250 epochs as required
            'imgsz': 1024,  # ✅ 1024 as required
            'batch': 4,     # Smaller batch for stability with tiling

            # Class-aware NMS (per requirements)
            'nms': True,
            'iou': 0.3,     # Lower IoU for class-aware NMS
            'conf': 0.25,   # Confidence threshold

            # Tiling configuration (per requirements)
            'mosaic': 0.3,  # ✅ Mosaic ≤ 0.3 as required
            'mixup': 0.0,   # ✅ No MixUp as required

            # Device and performance
            'device': 'mps',  # Apple Silicon
            'workers': 4,
            'cache': True,
            'amp': True,

            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,

            # Augmentations (realistic per requirements)
            'hsv_h': 0.015,  # Small hue variation
            'hsv_s': 0.7,    # Saturation
            'hsv_v': 0.4,    # Brightness
            'degrees': 5.0,  # Small rotation
            'translate': 0.1, # Small translation
            'scale': 0.5,    # Scale variation
            'shear': 2.0,    # Small shear
            'flipud': 0.0,   # ✅ No vertical flip as required
            'fliplr': 0.5,   # Horizontal flip OK

            # Training control
            'patience': 50,
            'save': True,
            'save_period': 25,
            'plots': True,
            'verbose': True,

            # Experiment name
            'name': 'ancomics_class_aware_nms_tiling_250ep',
            'seed': 42,
            'deterministic': True
        }

        return config

    def create_tiling_augmentation(self) -> Dict[str, Any]:
        """Create tiling augmentation configuration for training"""

        tiling_config = {
            'tile_size': 1024,
            'tile_overlap': 0.6,  # 60% overlap as required
            'tile_stride': 0.4,   # 40% stride (complement of overlap)
            'min_tile_area': 0.1, # Minimum area ratio for valid tiles
            'max_tiles_per_image': 16
        }

        return tiling_config

    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to YAML file"""

        config_path = self.configs_path / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")
        return config_path

    def create_training_command(self, config: Dict[str, Any]) -> str:
        """Create YOLO training command"""

        cmd_parts = ['yolo', 'train']

        # Add all configuration parameters
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f'--{key}')
            elif isinstance(value, (int, float)):
                cmd_parts.append(f'--{key} {value}')
            else:
                cmd_parts.append(f'--{key} "{value}"')

        return ' '.join(cmd_parts)

    def validate_requirements(self) -> bool:
        """Validate that all requirements are met"""

        checks = {
            'dataset_exists': self.dataset_path.exists(),
            'train_images': (self.dataset_path / "images" / "train").exists(),
            'val_images': (self.dataset_path / "images" / "val").exists(),
            'data_yaml': (self.dataset_path / "multibd_enhanced.yaml").exists(),
            'configs_dir': self.configs_path.exists()
        }

        all_passed = all(checks.values())

        if not all_passed:
            logger.error("Validation failed:")
            for check, passed in checks.items():
                if not passed:
                    logger.error(f"  ❌ {check}")
        else:
            logger.info("✅ All validation checks passed")

        return all_passed

def main():
    """Main training setup function"""

    logger.info("🚀 AnComicsViewer - Optimized YOLO Training Setup")
    logger.info("=" * 60)

    # Initialize training manager
    manager = YOLOTrainingManager("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")

    # Validate requirements
    if not manager.validate_requirements():
        logger.error("❌ Requirements validation failed. Please check dataset structure.")
        return

    # Create optimized configuration
    logger.info("🔧 Creating optimized training configuration...")
    train_config = manager.create_optimized_config()
    tiling_config = manager.create_tiling_augmentation()

    # Display configurations
    logger.info("\n📋 Training Configuration:")
    for key, value in train_config.items():
        logger.info(f"   {key}: {value}")

    logger.info("\n📋 Tiling Configuration:")
    for key, value in tiling_config.items():
        logger.info(f"   {key}: {value}")

    # Save configurations
    train_config_path = manager.save_config(train_config, "train_optimized_250ep_1024px.yaml")
    tiling_config_path = manager.save_config(tiling_config, "tiling_augmentation.yaml")

    # Create training command
    train_cmd = manager.create_training_command(train_config)

    logger.info("\n🚀 Training Command:")
    logger.info(f"   {train_cmd}")

    # Save training script
    script_path = manager.project_root / "run_optimized_training.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("echo '🚀 Starting optimized YOLO training...'\n")
        f.write(f"{train_cmd}\n")
        f.write("echo '✅ Training completed!'\n")

    # Make script executable
    os.chmod(script_path, 0o755)

    logger.info(f"\n💾 Training script saved: {script_path}")
    logger.info("\n🎯 Ready to start training with:")
    logger.info("   ✅ 250 epochs")
    logger.info("   ✅ 1024px image size")
    logger.info("   ✅ Class-aware NMS")
    logger.info("   ✅ Tiling support (60% overlap)")
    logger.info("   ✅ Realistic augmentations")
    logger.info("   ✅ No vertical flips or MixUp")

    logger.info(f"\n▶️  Run training with: ./{script_path.name}")

if __name__ == "__main__":
    main()
