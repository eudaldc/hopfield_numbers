"""Collate functions for DataLoader.

Because batches can mix tasks and have optional keys, collation
groups samples by task and stacks tensors within each group.
"""

import torch


# Keys that should stay as Python lists (not stacked into tensors)
_STRING_KEYS = {"task", "trial_id", "cohort", "congruency", "target_type"}

# Keys that are image tensors (variable presence, need stacking)
_IMAGE_KEYS = {
    "image",
    "left_image",
    "right_image",
    "sample_image",
    "target_image",
}

# Keys that are feature vectors
_FEATURE_KEYS = {
    "features",
    "left_features",
    "right_features",
    "sample_features",
    "target_features",
}


def _stack_or_list(key: str, values: list):
    """Stack numeric values into a tensor, keep strings as lists."""
    if key in _STRING_KEYS:
        return values

    if key in _IMAGE_KEYS or key in _FEATURE_KEYS:
        return torch.stack(values)

    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    if isinstance(first, (int, float)):
        return torch.tensor(values)
    return values


def multitask_collate(batch: list[dict]) -> dict[str, dict]:
    """Collate a heterogeneous batch into per-task sub-batches.

    Returns a dict keyed by task name. Each value is a dict of
    stacked tensors / lists for that task's samples in this batch.

    Example output structure::

        {
            "symbolic_comparison": {
                "subject_idx": tensor([...]),
                "left_digit": tensor([...]),
                ...
            },
            "dot_verification": {
                "subject_idx": tensor([...]),
                ...
            },
        }
    """
    by_task: dict[str, list[dict]] = {}
    for sample in batch:
        task = sample["task"]
        by_task.setdefault(task, []).append(sample)

    result = {}
    for task, samples in by_task.items():
        all_keys = set()
        for s in samples:
            all_keys.update(s.keys())

        collated = {}
        for key in all_keys:
            values = [s[key] for s in samples if key in s]
            if len(values) == len(samples):
                collated[key] = _stack_or_list(key, values)
            else:
                collated[key] = _stack_or_list(key, values)
        result[task] = collated

    return result


def single_task_collate(batch: list[dict]) -> dict:
    """Collate a single-task batch (all samples have the same task)."""
    all_keys = set()
    for s in batch:
        all_keys.update(s.keys())

    collated = {}
    for key in all_keys:
        values = [s[key] for s in batch if key in s]
        collated[key] = _stack_or_list(key, values)
    return collated
