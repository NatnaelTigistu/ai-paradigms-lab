"""Knowledge distillation loss, student-teacher transfer modules, and experiment runners."""

from paradigms.distillation.experiment import (
    effective_epochs,
    load_teacher,
    run_group,
    run_student,
)
from paradigms.distillation.loss import (
    distillation_loss_from_logits,
    make_distillation_loss,
)

__all__ = [
    "distillation_loss_from_logits",
    "make_distillation_loss",
    "effective_epochs",
    "load_teacher",
    "run_student",
    "run_group",
]
