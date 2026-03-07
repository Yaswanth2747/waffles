# models.py
# Purpose:
#   Load the neural network
#   Ensure the architecture is easy to split
#   Provide metadata (number of blocks)
# For the project, we use MobileNetV2, since it is lightweight and structured as sequential feature blocks.

import torch
import torchvision.models as models


SUPPORTED_MODELS = ("vit_b_16", "mobilenet_v2")


def load_model(model_name="vit_b_16", device="cpu"):
    """
    Load a supported pretrained model for inference.
    """

    if model_name == "vit_b_16":
        model = models.vit_b_16(weights="IMAGENET1K_V1")
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    model.eval()
    model.to(device)

    return model


def load_model2(device="cpu"):
    """
    Backward-compatible alias for MobileNetV2 loading.
    """

    return load_model(model_name="mobilenet_v2", device=device)

def get_feature_blocks(model):
    """
    Returns the list of feature blocks used for splitting.
    """
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return list(model.encoder.layers)
    if hasattr(model, "features"):
        return list(model.features)
    raise AttributeError("Model does not expose split-friendly blocks")


def get_num_splits(model):
    """
    Number of valid split points in the network.
    """
    return len(get_feature_blocks(model))


def get_input_tensor(device="cpu"):
    """
    Generate dummy input tensor for testing.
    """

    return torch.randn(1, 3, 224, 224).to(device)
