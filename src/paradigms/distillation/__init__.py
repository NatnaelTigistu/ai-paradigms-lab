"""Knowledge distillation loss and student-teacher transfer modules."""

from paradigms.distillation.loss import (
    distillation_loss_from_logits,
    make_distillation_loss,
)

__all__ = [
    "distillation_loss_from_logits",
    "make_distillation_loss",
]
