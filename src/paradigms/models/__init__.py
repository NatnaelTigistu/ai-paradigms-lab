"""Model architectures and model construction registry."""

from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.models.small_cnn import SmallCNN
from paradigms.models.registry import build_model, count_parameters

__all__ = [
    "TeacherCNN",
    "StudentMLP",
    "SmallCNN",
    "build_model",
    "count_parameters",
]
