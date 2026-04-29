"""Cross-cutting consistency tests.

Guards against stale references, key mismatches, and config/checkpoint
drift that can silently survive refactors.  Each test targets a specific
class of bug that is easy to introduce when renaming presets, output
keys, or checkpoint formats.
"""

import ast
import re
import tempfile
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.common.config import (
    ATTRACTOR_PRESETS,
    PRESET_FAMILIES,
    ExperimentConfig,
    load_config,
)
from src.common.io import (
    best_checkpoint_filename,
    infer_mode,
    infer_preset,
    save_checkpoint,
)
from src.training.engine import build_modules, build_task_heads

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# 1. PRESET_FAMILIES is the single source of truth
# =====================================================================

EXPECTED_PRESETS = {"default", "weber"}
EXPECTED_FAMILY = "attractor"
VALID_MODES = {"dynamic", "static"}


class TestPresetConsistency:
    """Every preset-related constant must agree with PRESET_FAMILIES."""

    def test_preset_families_keys(self):
        assert set(PRESET_FAMILIES.keys()) == EXPECTED_PRESETS

    def test_all_families_are_attractor(self):
        for preset, fam in PRESET_FAMILIES.items():
            assert fam == EXPECTED_FAMILY, f"preset {preset!r} has family {fam!r}"

    def test_attractor_presets_match(self):
        assert set(ATTRACTOR_PRESETS.keys()) == EXPECTED_PRESETS

    def test_experiment_config_default_preset_is_valid(self):
        cfg = ExperimentConfig()
        assert cfg.preset in PRESET_FAMILIES

    def test_experiment_config_family_property(self):
        for preset in PRESET_FAMILIES:
            cfg = ExperimentConfig(preset=preset)
            assert cfg.family == EXPECTED_FAMILY

    def test_no_legacy_fields_on_experiment_config(self):
        field_names = {f.name for f in fields(ExperimentConfig)}
        assert "ans" not in field_names, "ExperimentConfig still has an 'ans' field"
        assert "accumulator" not in field_names, "ExperimentConfig still has an 'accumulator' field"

    def test_build_modules_rejects_unknown_presets(self):
        for bad_preset in ("ans", "accumulator", "bogus"):
            cfg = ExperimentConfig(preset=bad_preset)
            with pytest.raises((ValueError, KeyError)):
                build_modules(cfg, 1, torch.device("cpu"))


# =====================================================================
# 2. Checkpoint naming encodes both preset and mode
# =====================================================================

class TestCheckpointNaming:

    def test_filename_includes_mode(self):
        for preset in EXPECTED_PRESETS:
            for mode in VALID_MODES:
                name = best_checkpoint_filename(preset, mode)
                assert preset in name
                assert mode in name
                assert name.endswith(".pt")

    def test_default_mode_is_dynamic(self):
        name = best_checkpoint_filename("default")
        assert "dynamic" in name

    def test_different_modes_produce_different_filenames(self):
        for preset in EXPECTED_PRESETS:
            a = best_checkpoint_filename(preset, "dynamic")
            b = best_checkpoint_filename(preset, "static")
            assert a != b, f"dynamic and static produce same filename for {preset}"

    def test_all_preset_mode_combos_are_unique(self):
        names = set()
        for preset in EXPECTED_PRESETS:
            for mode in VALID_MODES:
                name = best_checkpoint_filename(preset, mode)
                assert name not in names, f"duplicate checkpoint filename: {name}"
                names.add(name)


# =====================================================================
# 3. Checkpoint round-trip: save metadata → infer it back
# =====================================================================

class _Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.w


class TestCheckpointRoundTrip:

    @pytest.fixture(params=list(EXPECTED_PRESETS))
    def preset(self, request):
        return request.param

    @pytest.fixture(params=list(VALID_MODES))
    def mode(self, request):
        return request.param

    def test_infer_preset_from_extra(self, preset, mode):
        with tempfile.TemporaryDirectory() as tmp:
            model = _Dummy()
            path = Path(tmp) / best_checkpoint_filename(preset, mode)
            save_checkpoint(
                path, model, epoch=0,
                extra={"preset": preset, "attractor_mode": mode},
            )
            ckpt = torch.load(path, weights_only=False)
            extra = ckpt.get("extra", {})
            assert infer_preset(ckpt["model_state_dict"], extra) == preset
            assert infer_mode(extra) == mode

    def test_infer_defaults_without_extra(self):
        assert infer_preset({}, None) == "default"
        assert infer_mode(None) == "dynamic"
        assert infer_mode({}) == "dynamic"


# =====================================================================
# 4. Config defaults match the attractor-first framing
# =====================================================================

class TestConfigDefaults:

    def test_attractor_mode_default_is_dynamic(self):
        cfg = ExperimentConfig()
        assert cfg.attractor_mode == "dynamic"

    def test_yaml_config_attractor_mode_is_dynamic(self):
        cfg = load_config()
        assert cfg.attractor_mode == "dynamic"

    def test_yaml_config_preset_is_valid(self):
        cfg = load_config()
        assert cfg.preset in PRESET_FAMILIES

    def test_yaml_has_no_legacy_sections(self):
        import yaml
        with open(_PROJECT_ROOT / "configs" / "config.yaml") as f:
            raw = yaml.safe_load(f) or {}
        assert "ans" not in raw, "config.yaml still has top-level 'ans' section"
        assert "accumulator" not in raw, "config.yaml still has top-level 'accumulator' section"
        assert "frontend_family" not in raw.get("synthetic", {}), \
            "config.yaml still has 'frontend_family'"


# =====================================================================
# 5. DynamicAttractorCore output keys use y0, not y
# =====================================================================

class TestOutputKeys:

    def test_dynamic_core_uses_y0_key(self):
        from src.models.attractor_dynamics import DynamicAttractorCore

        for cleanup_mode in ("gradient", "none"):
            core = DynamicAttractorCore(
                support_min=1, support_max=4, cleanup_mode=cleanup_mode,
            )
            y = torch.tensor([1.5, 2.5, 3.5])
            out = core.cleanup(y)
            assert "y0" in out, f"cleanup_mode={cleanup_mode!r}: missing 'y0' key"
            assert "y" not in out, f"cleanup_mode={cleanup_mode!r}: stale 'y' key present"


# =====================================================================
# 6. discover_checkpoints key format matches consumer loops
#    (this is the exact pattern that caused the original bug)
# =====================================================================

class TestDiscoverCheckpointsContract:
    """Verify that keys from discover_checkpoints() can be used as
    dict keys everywhere in the eval loop without KeyError."""

    def test_discover_keys_are_preset_slash_mode(self, tmp_path):
        from scripts.eval_core import discover_checkpoints, CHECKPOINT_DIR

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        model = _Dummy()
        save_checkpoint(
            ckpt_dir / best_checkpoint_filename("default", "dynamic"),
            model, extra={"preset": "default", "attractor_mode": "dynamic"},
        )
        save_checkpoint(
            ckpt_dir / best_checkpoint_filename("weber", "static"),
            model, extra={"preset": "weber", "attractor_mode": "static"},
        )

        import scripts.eval_core as ec
        orig = ec.CHECKPOINT_DIR
        try:
            ec.CHECKPOINT_DIR = ckpt_dir
            found = ec.discover_checkpoints()
        finally:
            ec.CHECKPOINT_DIR = orig

        for key in found:
            parts = key.split("/")
            assert len(parts) == 2, f"key {key!r} is not 'preset/mode'"
            preset, mode = parts
            assert preset in EXPECTED_PRESETS, f"unknown preset in key: {key!r}"
            assert mode in VALID_MODES, f"unknown mode in key: {key!r}"

    def test_results_dict_uses_combo_key_consistently(self):
        """Parse eval_core.py and verify that after the for-loop assigns to
        results[combo], all subsequent accesses within the loop body also
        use combo (not preset)."""
        source = (_PROJECT_ROOT / "scripts" / "eval_core.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            target = node.target
            if not isinstance(target, ast.Tuple):
                continue
            elts = target.elts
            if len(elts) != 2:
                continue
            loop_var = elts[0]
            if not isinstance(loop_var, ast.Name):
                continue
            if loop_var.id != "combo":
                continue

            # Found the for-loop: `for combo, ckpt_path in ...`
            # Every `results[X]` inside this loop must use `combo` as key
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript):
                    if (isinstance(child.value, ast.Name)
                            and child.value.id == "results"):
                        key = child.slice
                        if isinstance(key, ast.Name):
                            assert key.id == "combo", (
                                f"results[{key.id}] at line {child.lineno} — "
                                f"should be results[combo]"
                            )
            return

        pytest.fail("Could not find 'for combo, ckpt_path in ...' loop in eval_core.py")


# =====================================================================
# 7. Scripts and docstrings reference only valid presets
# =====================================================================

_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_LEGACY_PRESET_RE = re.compile(
    r"""--preset\s+(ans|accumulator)"""
    r"""|best_(ans|accumulator)\.pt"""
    r"""|"(ans|accumulator)"\s*:\s*"(ans|accumulator)"\s*""",
)

# Files that intentionally discuss legacy baselines
_LEGACY_ALLOWLIST = {
    "run_regime_recovery.py",
}


class TestNoLegacyPresetInScripts:

    @pytest.fixture(params=sorted(
        p.name for p in _SCRIPTS_DIR.glob("*.py")
        if p.name not in _LEGACY_ALLOWLIST
    ))
    def script_name(self, request):
        return request.param

    def test_no_legacy_preset_in_script(self, script_name):
        text = (_SCRIPTS_DIR / script_name).read_text()
        match = _LEGACY_PRESET_RE.search(text)
        assert match is None, (
            f"{script_name}: legacy preset reference at "
            f"position {match.start()}: {match.group()!r}"
        )


_SRC_DIR = _PROJECT_ROOT / "src"
_OLD_CHECKPOINT_RE = re.compile(r"best_(default|weber)\.pt")

# Files where old patterns are OK (self-contained legacy modules)
_SRC_ALLOWLIST = {
    "baseline_ans.py",
    "baseline_accumulator.py",
    "legacy_frontends.py",
    "effective_baselines.py",
}


class TestNoOldCheckpointPatternInSrc:

    @pytest.fixture(params=sorted(
        str(p.relative_to(_SRC_DIR))
        for p in _SRC_DIR.rglob("*.py")
        if p.name not in _SRC_ALLOWLIST
    ))
    def src_rel(self, request):
        return request.param

    def test_no_old_checkpoint_pattern(self, src_rel):
        text = (_SRC_DIR / src_rel).read_text()
        match = _OLD_CHECKPOINT_RE.search(text)
        assert match is None, (
            f"src/{src_rel}: old checkpoint pattern {match.group()!r} "
            f"(should include mode, e.g. best_default_dynamic.pt)"
        )
