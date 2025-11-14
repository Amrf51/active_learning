"""Generate multiple experiment configurations automatically."""

import argparse
import logging
from pathlib import Path
from itertools import product
import json

from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_experiments(
    base_config_path: str,
    output_dir: str = "experiments",
    params_grid: dict = None
) -> list:
    """Generate multiple experiment configs from parameter grid.
    
    Args:
        base_config_path: Path to base config YAML
        output_dir: Directory to save generated configs
        params_grid: Dictionary of {parameter_path: [values]}
                    Example: {
                        "model.name": ["resnet18", "resnet50"],
                        "training.learning_rate": [0.001, 0.0001],
                        "training.optimizer": ["adam", "sgd"]
                    }
    
    Returns:
        List of paths to generated config files
    """
    if params_grid is None:
        params_grid = {
            "model.name": ["resnet18", "resnet50", "mobilenetv2_100"],
            "training.learning_rate": [0.001],
            "training.optimizer": ["adam"],
        }
    
    # Load base config
    logger.info(f"Loading base config from: {base_config_path}")
    base_config = Config.from_yaml(base_config_path)
    
    # Create output directory
    experiments_dir = Path(output_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Generating {len(combinations)} experiment configurations...")
    
    generated_configs = []
    
    for idx, combo in enumerate(combinations, 1):
        # Create experiment folder
        exp_name = f"exp_{idx:03d}"
        exp_dir = experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy base config
        config = Config.from_yaml(base_config_path)
        
        # Apply parameter values
        param_dict = {}
        for param_name, param_value in zip(param_names, combo):
            param_dict[param_name] = param_value
        
        # Update config with parameters
        config = _update_config_from_dict(config, param_dict)
        
        # Save config
        config_path = exp_dir / "config.yaml"
        config.save_to(str(config_path))
        generated_configs.append(config_path)
        
        # Log experiment
        params_str = ", ".join(
            [f"{k}={v}" for k, v in zip(param_names, combo)]
        )
        logger.info(f"[{idx}/{len(combinations)}] {exp_name}: {params_str}")
    
    logger.info(f"✅ Generated {len(generated_configs)} configs in {experiments_dir}")
    
    return generated_configs


def _update_config_from_dict(config: Config, params_dict: dict) -> Config:
    """Update config object from nested parameter dictionary.
    
    Args:
        config: Config object
        params_dict: Dictionary like {"model.name": "resnet50", "training.lr": 0.001}
    
    Returns:
        Updated config object
    """
    for key, value in params_dict.items():
        parts = key.split(".")
        
        # Navigate to parent object
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Set the value
        setattr(obj, parts[-1], value)
    
    return config


def print_generated_summary(config_paths: list):
    """Print summary of generated configs.
    
    Args:
        config_paths: List of config file paths
    """
    print("\n" + "="*60)
    print("Generated Experiments Summary")
    print("="*60)
    
    for i, config_path in enumerate(config_paths, 1):
        config = Config.from_yaml(str(config_path))
        exp_dir = config_path.parent.name
        print(f"\n[{i}] {exp_dir}/")
        print(f"    Model: {config.model.name}")
        print(f"    Learning Rate: {config.training.learning_rate}")
        print(f"    Optimizer: {config.training.optimizer}")
        print(f"    Batch Size: {config.training.batch_size}")
    
    print("\n" + "="*60)
    print(f"Total: {len(config_paths)} experiments")
    print("="*60)
    print("\nTo run all experiments:")
    print("  for config in experiments/exp_*/config.yaml; do")
    print(f"    python train.py --config $config --exp-name $(basename $(dirname $config))")
    print("  done")
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate multiple experiment configurations"
    )
    parser.add_argument(
        "--base-config",
        default="config/base_config.yaml",
        help="Path to base config template"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Output directory for generated configs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "resnet50"],
        help="List of model names to test"
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001],
        help="List of learning rates to test"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["adam"],
        help="List of optimizers to test"
    )
    parser.add_argument(
        "--sampling-strategies",
        nargs="+",
        default=[],
        help="List of AL sampling strategies (optional)"
    )
    
    args = parser.parse_args()
    
    # Build parameter grid
    params_grid = {
        "model.name": args.models,
        "training.learning_rate": args.learning_rates,
        "training.optimizer": args.optimizers,
    }
    
    # Add AL strategies if provided
    if args.sampling_strategies:
        params_grid["active_learning.sampling_strategy"] = args.sampling_strategies
    
    # Generate configs
    config_paths = generate_experiments(
        args.base_config,
        args.output_dir,
        params_grid
    )
    
    # Print summary
    print_generated_summary(config_paths)


if __name__ == "__main__":
    main()