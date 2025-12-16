#!/usr/bin/env python3
"""
Configuration Management CLI Tool

This tool helps manage experiment configurations and templates.

Usage:
    # List all templates
    python manage_configs.py list-templates
    
    # Create new experiment config
    python manage_configs.py create-experiment my_experiment --template base_config --model.name resnet50
    
    # Save experiment config as template
    python manage_configs.py save-template experiments/my_exp_20241216_143022 --name my_custom_template
    
    # Edit experiment config
    python manage_configs.py edit-config experiments/my_exp_20241216_143022 --training.learning_rate 0.0001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config, ConfigManager


def list_templates(args):
    """List all available config templates."""
    manager = ConfigManager()
    templates = manager.list_templates()
    
    if not templates:
        print("No templates found.")
        return
    
    print(f"\n📋 Available Config Templates ({len(templates)} found)")
    print("=" * 60)
    
    for template in templates:
        status = "🔄 AL" if template["has_active_learning"] else "📊 Standard"
        legacy = " (Legacy)" if template["is_legacy"] else ""
        
        print(f"\n{status} {template['name']}{legacy}")
        print(f"   File: {template['filename']}")
        print(f"   Description: {template['description']}")
        if template['tags']:
            print(f"   Tags: {', '.join(template['tags'])}")
        print(f"   Created: {template['created_at']}")


def create_experiment(args):
    """Create a new experiment with custom config."""
    manager = ConfigManager()
    
    # Parse overrides from command line
    overrides = {}
    if hasattr(args, 'overrides') and args.overrides:
        for override in args.overrides:
            if '=' not in override:
                print(f"❌ Invalid override format: {override}")
                print("   Use format: key=value (e.g., model.name=resnet50)")
                return
            
            key, value = override.split('=', 1)
            
            # Try to parse value as appropriate type
            try:
                # Try int first
                if value.isdigit():
                    value = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                # Try boolean
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # Keep as string
            except ValueError:
                pass
            
            overrides[key] = value
    
    try:
        exp_dir, config = manager.create_experiment_config(
            experiment_name=args.name,
            base_template=args.template,
            **overrides
        )
        
        print(f"✅ Created experiment: {exp_dir.name}")
        print(f"   Directory: {exp_dir}")
        print(f"   Config: {exp_dir}/config.yaml")
        print(f"   Template: {args.template}")
        
        if overrides:
            print(f"   Overrides applied:")
            for key, value in overrides.items():
                print(f"     {key} = {value}")
        
        print(f"\n🚀 To run this experiment:")
        print(f"   python train.py --config {exp_dir}/config.yaml --exp-name {exp_dir.name}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Available templates:")
        templates = manager.list_templates()
        for t in templates:
            print(f"     - {t['name']}")


def save_template(args):
    """Save an experiment's config as a new template."""
    manager = ConfigManager()
    exp_dir = Path(args.experiment_dir)
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    try:
        template_path = manager.copy_config_as_template(
            experiment_dir=exp_dir,
            template_name=args.name,
            description=args.description or f"Template created from {exp_dir.name}"
        )
        
        print(f"✅ Saved template: {args.name}")
        print(f"   File: {template_path}")
        print(f"   Source: {exp_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")


def edit_config(args):
    """Edit an experiment's configuration."""
    manager = ConfigManager()
    exp_dir = Path(args.experiment_dir)
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    # Parse updates from command line
    updates = {}
    if hasattr(args, 'updates') and args.updates:
        for update in args.updates:
            if '=' not in update:
                print(f"❌ Invalid update format: {update}")
                print("   Use format: key=value (e.g., training.learning_rate=0.0001)")
                return
            
            key, value = update.split('=', 1)
            
            # Try to parse value as appropriate type
            try:
                if value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
            except ValueError:
                pass
            
            updates[key] = value
    
    if not updates:
        print("❌ No updates specified")
        return
    
    try:
        config = manager.update_experiment_config(exp_dir, **updates)
        
        print(f"✅ Updated config: {exp_dir}/config.yaml")
        print(f"   Changes:")
        for key, value in updates.items():
            print(f"     {key} = {value}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")


def show_config(args):
    """Show an experiment's configuration."""
    exp_dir = Path(args.experiment_dir)
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        config = Config.from_yaml(str(config_path))
        config_dict = config.to_dict()
        
        print(f"\n📄 Configuration: {exp_dir.name}")
        print("=" * 60)
        print(json.dumps(config_dict, indent=2))
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")


def compare_configs(args):
    """Compare two experiment configurations."""
    exp1_dir = Path(args.experiment1)
    exp2_dir = Path(args.experiment2)
    
    for exp_dir in [exp1_dir, exp2_dir]:
        if not exp_dir.exists():
            print(f"❌ Experiment directory not found: {exp_dir}")
            return
    
    try:
        config1 = Config.from_yaml(str(exp1_dir / "config.yaml"))
        config2 = Config.from_yaml(str(exp2_dir / "config.yaml"))
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        print(f"\n🔍 Comparing Configurations")
        print("=" * 60)
        print(f"Experiment 1: {exp1_dir.name}")
        print(f"Experiment 2: {exp2_dir.name}")
        print()
        
        differences = _find_differences(dict1, dict2)
        
        if not differences:
            print("✅ Configurations are identical")
        else:
            print(f"Found {len(differences)} differences:")
            for key, (val1, val2) in differences.items():
                print(f"  {key}:")
                print(f"    {exp1_dir.name}: {val1}")
                print(f"    {exp2_dir.name}: {val2}")
        
    except Exception as e:
        print(f"❌ Error comparing configs: {e}")


def _find_differences(dict1: Dict[str, Any], dict2: Dict[str, Any], prefix: str = "") -> Dict[str, tuple]:
    """Find differences between two nested dictionaries."""
    differences = {}
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        
        if key not in dict1:
            differences[full_key] = (None, dict2[key])
        elif key not in dict2:
            differences[full_key] = (dict1[key], None)
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # Recursively compare nested dicts
            nested_diffs = _find_differences(dict1[key], dict2[key], full_key)
            differences.update(nested_diffs)
        elif dict1[key] != dict2[key]:
            differences[full_key] = (dict1[key], dict2[key])
    
    return differences


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage experiment configurations and templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List templates
    list_parser = subparsers.add_parser('list-templates', help='List all config templates')
    list_parser.set_defaults(func=list_templates)
    
    # Create experiment
    create_parser = subparsers.add_parser('create-experiment', help='Create new experiment config')
    create_parser.add_argument('name', help='Experiment name')
    create_parser.add_argument('--template', default='base_config', help='Base template to use')
    create_parser.add_argument('overrides', nargs='*', help='Config overrides (key=value)')
    create_parser.set_defaults(func=create_experiment)
    
    # Save template
    save_parser = subparsers.add_parser('save-template', help='Save experiment config as template')
    save_parser.add_argument('experiment_dir', help='Path to experiment directory')
    save_parser.add_argument('--name', required=True, help='Template name')
    save_parser.add_argument('--description', help='Template description')
    save_parser.set_defaults(func=save_template)
    
    # Edit config
    edit_parser = subparsers.add_parser('edit-config', help='Edit experiment configuration')
    edit_parser.add_argument('experiment_dir', help='Path to experiment directory')
    edit_parser.add_argument('updates', nargs='*', help='Config updates (key=value)')
    edit_parser.set_defaults(func=edit_config)
    
    # Show config
    show_parser = subparsers.add_parser('show-config', help='Show experiment configuration')
    show_parser.add_argument('experiment_dir', help='Path to experiment directory')
    show_parser.set_defaults(func=show_config)
    
    # Compare configs
    compare_parser = subparsers.add_parser('compare-configs', help='Compare two experiment configs')
    compare_parser.add_argument('experiment1', help='First experiment directory')
    compare_parser.add_argument('experiment2', help='Second experiment directory')
    compare_parser.set_defaults(func=compare_configs)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()