from pathlib import Path

import pytest
import yaml

from drmc_rl.program.model import ProgramSpec, RecipeSpec


def test_program_manifest_loads_and_references_known_gates() -> None:
    path = Path(__file__).parents[1] / "drmc_rl" / "program" / "program.yaml"
    spec = ProgramSpec.load(path)
    assert spec.products["human-trainer"].competitive_core == "unified-g5"
    assert spec.recipes["g4-strong-league"].status == "complete"
    assert spec.recipes["g4-strong-league-rewarm"].status == "complete"
    assert spec.recipes["g4-strong-league-rewarm-900m"].status == "complete"
    assert spec.recipes["human-afterstate-bootstrap"].status == "complete"
    assert spec.gates["pair-state-v2"].status == "complete"
    spec.validate()


def test_recipe_command_rejects_missing_substitution() -> None:
    recipe = RecipeSpec.from_mapping(
        "x",
        {
            "status": "staged",
            "stage": 1,
            "purpose": "test",
            "command": ["python", "{missing}"],
        },
    )
    with pytest.raises(ValueError, match="missing"):
        recipe.resolved_command(".")


def test_gate_cycle_is_rejected(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "gates": {
            "a": {"depends_on": ["b"]},
            "b": {"depends_on": ["a"]},
        },
        "recipes": {},
        "products": {},
    }
    path = tmp_path / "program.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="cycle"):
        ProgramSpec.load(path)


def test_launch_options_work_after_recipe_name() -> None:
    from tools.program import build_parser

    args = build_parser().parse_args(
        ["launch", "human-afterstate-bootstrap", "--dry-run", "--set", "dataset=/data"]
    )
    assert args.recipe == "human-afterstate-bootstrap"
    assert args.dry_run
    assert args.set == ["dataset=/data"]
