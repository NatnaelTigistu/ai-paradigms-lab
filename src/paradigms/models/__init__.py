"""Model architectures and model construction registry."""

from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.models.registry import build_model, count_parameters

__all__ = [
    "TeacherCNN",
    "StudentMLP",
    "build_model",
    "count_parameters",
]
