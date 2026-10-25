"""Tests for teacher CNN, student MLP, model registry, and parameter counting."""

from __future__ import annotations

import pytest
import torch

from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.models.registry import build_model, count_parameters


def test_forward_output_shapes() -> None:
    """Verify both TeacherCNN and StudentMLP map (B, 1, 28, 28) inputs to (B, 10)."""
    x = torch.randn(4, 1, 28, 28)
    cnn = TeacherCNN()
    mlp = StudentMLP(hidden=32)

    cnn_out = cnn(x)
    mlp_out = mlp(x)

    assert cnn_out.shape == (4, 10)
    assert mlp_out.shape == (4, 10)


def test_teacher_cnn_parameter_count() -> None:
    """Verify TeacherCNN parameter count matches expected analytical sum (421,642)."""
    model = TeacherCNN()
    # conv1: (1 * 3 * 3 + 1) * 32 = 320
    # conv2: (32 * 3 * 3 + 1) * 64 = 18496
    # fc1: (3136 + 1) * 128 = 401536
    # fc2: (128 + 1) * 10 = 1290
    assert count_parameters(model) == 421642


def test_student_mlp_parameter_counts() -> None:
    """Verify StudentMLP parameter counts for default and swept hidden sizes."""
    mlp_32 = StudentMLP(hidden=32)
    # fc1: (784 + 1) * 32 = 25120
    # fc2: (32 + 1) * 10 = 330
    assert count_parameters(mlp_32) == 25450

    mlp_64 = StudentMLP(hidden=64)
    assert count_parameters(mlp_64) > count_parameters(mlp_32)


def test_outputs_are_raw_logits() -> None:
    """Verify outputs are unnormalized logits rather than probabilities (contain negatives, sum != 1)."""
    x = torch.randn(10, 1, 28, 28)
    models = [TeacherCNN(), StudentMLP(hidden=32)]

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(x)

        # Output contains negative values (softmax probabilities are strictly in [0, 1])
        assert (out < 0).any(), f"{model.__class__.__name__} should output negative values for unnormalized logits"

        # Row sums are not 1.0
        row_sums = out.sum(dim=-1)
        assert not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2
        ), f"{model.__class__.__name__} row sums should not sum to 1.0"


def test_teacher_cnn_determinism_in_eval_and_stochastic_in_train() -> None:
    """Verify model.eval() disables dropout for deterministic inference, while train() differs."""
    cnn = TeacherCNN(dropout=0.5)
    x = torch.randn(16, 1, 28, 28)

    # eval mode: deterministic forward passes
    cnn.eval()
    with torch.no_grad():
        out1 = cnn(x)
        out2 = cnn(x)
    assert torch.equal(out1, out2)

    # train mode: stochastic forward passes due to dropout
    cnn.train()
    train_out1 = cnn(x)
    train_out2 = cnn(x)
    assert not torch.equal(train_out1, train_out2)


def test_registry_build_model_valid() -> None:
    """Verify build_model instantiates registered architectures with given arguments."""
    teacher = build_model("teacher_cnn", dropout=0.2)
    assert isinstance(teacher, TeacherCNN)

    student = build_model("student_mlp", hidden=64)
    assert isinstance(student, StudentMLP)
    assert student.hidden == 64


def test_registry_build_model_unknown_raises() -> None:
    """Verify build_model raises ValueError listing valid names for unknown architectures."""
    with pytest.raises(ValueError, match="Unknown model name 'bogus'"):
        build_model("bogus")
