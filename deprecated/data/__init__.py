"""Data loading, schemas, dataset builders, and synthetic benchmarks."""

from src.data.synth_benchmarks import (
    NeutralEvidenceChannel,
    generate_task_a,
    generate_task_b,
    generate_task_c,
    build_benchmark_datasets,
)
from src.data.schemas import (
    FEATURE_COLUMNS,
    SCHEMA_REGISTRY,
    TASK_NAMES,
)
from src.data.splits import filter_by_subjects, subject_split
from src.data.dataset_builders import (
    SubjectIndexMap,
    SymbolicComparisonDataset,
    MatchToSampleDataset,
    DotVerificationDataset,
    PanamathDataset,
    MultitaskDataset,
    MultitaskSampler,
    DATASET_REGISTRY,
    build_dataloader,
)
from src.data.collate import multitask_collate, single_task_collate
from src.data.stimulus_features import (
    extract_features,
    extract_features_from_path,
    build_stimulus_feature_table,
)

__all__ = [
    # synthetic benchmarks
    "NeutralEvidenceChannel",
    "generate_task_a",
    "generate_task_b",
    "generate_task_c",
    "build_benchmark_datasets",
    # schemas
    "FEATURE_COLUMNS",
    "SCHEMA_REGISTRY",
    "TASK_NAMES",
    # splits
    "subject_split",
    "filter_by_subjects",
    # datasets
    "SubjectIndexMap",
    "SymbolicComparisonDataset",
    "MatchToSampleDataset",
    "DotVerificationDataset",
    "PanamathDataset",
    "MultitaskDataset",
    "MultitaskSampler",
    "DATASET_REGISTRY",
    "build_dataloader",
    # collate
    "multitask_collate",
    "single_task_collate",
    # stimulus features
    "extract_features",
    "extract_features_from_path",
    "build_stimulus_feature_table",
]
