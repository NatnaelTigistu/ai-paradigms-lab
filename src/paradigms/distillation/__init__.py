"""Knowledge distillation loss, student-teacher transfer modules, and experiment runners."""

from paradigms.distillation.cached import (
    cache_teacher_logits,
    masked_distillation_loss,
)
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
from paradigms.distillation.v2 import run_student_v2

__all__ = [
    "distillation_loss_from_logits",
    "make_distillation_loss",
    "cache_teacher_logits",
    "masked_distillation_loss",
    "effective_epochs",
    "load_teacher",
    "run_student",
    "run_student_v2",
    "run_group",
]
