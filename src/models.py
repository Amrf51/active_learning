"""Model loading and architecture management using TIMM."""

import timm
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def get_model(config, device: str = "cpu") -> nn.Module:
    """Load model from TIMM library.
    
    Args:
        config: ModelConfig object with name, pretrained, num_classes
        device: Device to put model on ("cpu" or "cuda")
        
    Returns:
        PyTorch model on specified device
    """
    try:
        model = timm.create_model(
            config.name,
            pretrained=config.pretrained,
            num_classes=config.num_classes
        )
        logger.info(f"Loaded model: {config.name} (pretrained={config.pretrained}, "
                   f"num_classes={config.num_classes})")
    except Exception as e:
        logger.error(f"Failed to load model {config.name}: {e}")
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


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract feature embeddings from a model.

    Falls back to flattening model outputs when an explicit feature hook is
    unavailable. This helper is intended for sampling strategies that need a
    latent representation (e.g., diversity sampling with clustering).

    Args:
        model: PyTorch model
        images: Input batch tensor

    Returns:
        Feature tensor for downstream processing
    """
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
        if isinstance(features, (list, tuple)):
            features = features[-1]
    elif hasattr(model, "get_features") and callable(getattr(model, "get_features")):
        features = model.get_features(images)
    else:
        features = torch.flatten(model(images), 1)

    return features


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