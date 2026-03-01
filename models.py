"""Model loading and architecture management using TIMM."""

import timm
import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_model(name: str, num_classes: int, pretrained: bool = True, device: str = "cpu") -> nn.Module:
    """Load model from TIMM library.
    
    Args:
        name: Model name (e.g., "resnet50", "mobilenetv3_large_100")
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        device: Device to put model on ("cpu" or "cuda")
        
    Returns:
        PyTorch model on specified device
    """
    try:
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        logger.info(f"Loaded model: {name} (pretrained={pretrained}, "
                   f"num_classes={num_classes})")
    except Exception as e:
        logger.error(f"Failed to load model {name}: {e}")
        raise
    
    model = model.to(device)
    return model


def get_model_info(model: nn.Module) -> dict:
    """Get model information (parameters, etc.).
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model stats
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_name": model.__class__.__name__,
    }


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters in model (useful for transfer learning).
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Froze all model parameters")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters in model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfroze all model parameters")


def freeze_backbone_unfreeze_head(model: nn.Module, num_layers_to_unfreeze: int = 2) -> None:
    """Freeze backbone, unfreeze last N layers (common transfer learning strategy).
    
    Args:
        model: PyTorch model
        num_layers_to_unfreeze: Number of final layers to unfreeze
    """
    freeze_backbone(model)
    
    # Get layers (this is model-dependent, this is a general approach)
    layers = list(model.children())
    for layer in layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    logger.info(f"Froze backbone, unfroze last {num_layers_to_unfreeze} layers")


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Common TIMM model names for reference
AVAILABLE_ARCHITECTURES = {
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "mobilenet": ["mobilenetv2_100", "mobilenetv3_small_100", "mobilenetv3_large_100"],
    "efficientnet": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
    "vgg": ["vgg11", "vgg13", "vgg16", "vgg19"],
    "densenet": ["densenet121", "densenet169", "densenet201"],
}


def list_available_models() -> dict:
    """Print available model architectures.
    
    Returns:
        Dictionary of available architectures
    """
    return AVAILABLE_ARCHITECTURES


def search_timm_models(query: str = "", pretrained_only: bool = True) -> list:
    """Search TIMM model registry for UI dropdown.
    
    Args:
        query: Search term (e.g., "resnet", "mobile", "efficient")
        pretrained_only: Only show models with pretrained weights
        
    Returns:
        List of model name strings (capped at 50 results for UI performance)
    """
    pattern = f"*{query}*" if query else "*"
    results = timm.list_models(pattern, pretrained=pretrained_only)
    return results[:50]  # cap for UI dropdown performance


def get_model_families() -> dict:
    """Get curated model families for UI dropdown groups.
    
    Returns:
        Dictionary mapping family names to lists of model names
    """
    return {
        "ResNet (Recommended)": [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ],
        "MobileNet (Lightweight)": [
            "mobilenetv2_100", "mobilenetv3_small_100", "mobilenetv3_large_100"
        ],
        "EfficientNet": [
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"
        ],
        "DenseNet": [
            "densenet121", "densenet169", "densenet201"
        ],
        "VGG": [
            "vgg11", "vgg13", "vgg16", "vgg19"
        ],
    }


def get_model_card(model_name: str) -> dict:
    """Get info about a TIMM model for UI display.
    
    Useful for showing parameter count before the user commits to a model.
    
    Args:
        model_name: Name of the TIMM model
        
    Returns:
        Dictionary with model information:
        - name: Model name
        - parameters: Total parameter count
        - parameters_human: Human-readable parameter count (e.g., "25.6M")
        - has_pretrained: Whether pretrained weights are available
        - error: Error message if model info couldn't be retrieved
    """
    try:
        # Create a temporary model to get parameter count
        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        del model  # free memory immediately
        
        # Check if pretrained weights are available
        pretrained_models = timm.list_models(pretrained=True)
        has_pretrained = model_name in pretrained_models
        
        return {
            "name": model_name,
            "parameters": total_params,
            "parameters_human": f"{total_params / 1e6:.1f}M",
            "has_pretrained": has_pretrained,
        }
    except Exception as e:
        logger.warning(f"Could not get model card for {model_name}: {e}")
        return {
            "name": model_name,
            "error": str(e)
        }