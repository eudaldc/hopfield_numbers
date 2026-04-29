"""Model components: visual encoder, attractor core, task heads, baselines."""

from src.models.attractor_core import AttractorCore
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.baseline_accumulator import AccumulatorCore
from src.models.baseline_ans import ANSCore
from src.models.rt_proxy import RTProxy
from src.models.subject_params import SubjectParams, inverse_softplus
from src.models.symbolic_head import SymbolicHead
from src.models.task_heads import (
    DotVerificationHead,
    MatchToSampleHead,
    PanamathHead,
    SymbolicComparisonHead,
    TASK_HEAD_REGISTRY,
)
from src.models.visual_encoder import (
    FeatureEncoder,
    TinyImageEncoder,
    build_visual_encoder,
)

__all__ = [
    "AttractorCore",
    "DynamicAttractorCore",
    "ANSCore",
    "AccumulatorCore",
    "RTProxy",
    "SubjectParams",
    "inverse_softplus",
    "SymbolicHead",
    "SymbolicComparisonHead",
    "MatchToSampleHead",
    "DotVerificationHead",
    "PanamathHead",
    "TASK_HEAD_REGISTRY",
    "FeatureEncoder",
    "TinyImageEncoder",
    "build_visual_encoder",
]
