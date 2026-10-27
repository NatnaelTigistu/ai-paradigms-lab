"""Model registry and parameter accounting utilities."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.models.small_cnn import SmallCNN

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "teacher_cnn": TeacherCNN,
    "student_mlp": StudentMLP,
    "small_cnn": SmallCNN,
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model architecture from its registered identifier.

    Args:
        name: Name of the model architecture ("teacher_cnn" or "student_mlp").
        **kwargs: Architecture-specific hyperparameters forwarded to model constructor.

    Raises:
        ValueError: If `name` is not in the registry, providing the valid options.

    Returns:
        Instantiated nn.Module instance.
    """
    if name not in _REGISTRY:
        valid_names = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown model name '{name}'. Valid registered models are: {valid_names}."
        )
    return _REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module.

    Filters by `requires_grad` to accurately report capacity and exclude frozen
    layers or static buffers during transfer or evaluation.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
