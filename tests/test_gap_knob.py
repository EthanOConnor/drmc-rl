"""Skill-gap knobs (knobs-v2): features, sign, endgame gating, immediate rule and registry wiring."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.style import gap_knob as gk
from drmc_rl.style import knobs
from drmc_rl.style import showy_knob as sk

MODELS = Path(sk.__file__).parent / "models"
GAP_KEYS = [k for k in knobs.REGISTRY if k.startswith("gap-")]


def _field():
    f = np.full(128, 0xFF, np.uint8)
    f[120:124] = (0xD1, 0xD1, 0xD1, 0xFF)   # three red viruses on the floor
    f[112] = 0xD0                             # a yellow virus above the first
    return f


ACTIONS = np.array([0 * 128 + 115, 1 * 128 + 99, 0 * 128 + 96, 0 * 128 + 100, 0 * 128 + 116, -1])
MASK = np.array([True, True, True, True, True, False])


def _spec(names, coef, sign=1.0, **extra):
    n = len(names)
    return dict(schema=gk.SCHEMA, label="test", sign=sign, features=list(names), mean=[0.0] * n, scale=[1.0] * n,
                coef=list(coef), intercept=0.0, **extra)


def test_gap_features_count_what_they_name():
    f = np.full(128, 0xFF, np.uint8)
    f[15 * 8 + 0] = 0xD1                     # red virus bottom-left, on the edge, nothing below: not stranded
    f[5 * 8 + 7] = 0xD2                      # blue virus high on the right edge, 10 empty cells below: stranded
    f[14 * 8 + 0] = 0x60 | 0                 # a yellow pill half on the red virus: covers it, isolated
    f[2 * 8 + 3] = 0x40 | 1                  # a red half in the spawn zone (row 2, column 3)
    x = dict(zip(gk.GAP_NAMES, gk.gap_features(f[None])[0]))
    assert x["gap_viruses"] == 2 and x["gap_pill_cells"] == 2
    assert x["gap_covered_viruses"] == 1 and x["gap_buried"] == 1 and x["gap_exposed_viruses"] == 1
    assert x["gap_isolated"] == 2 and x["gap_edge_strand"] == 1
    assert x["gap_h34"] == 14 and x["gap_top3"] == 1 and x["gap_near_topout"] == 0


def test_sign_flips_the_bias_and_bias_is_centered():
    names = ["gap_covered_viruses", "holes"]
    up = gk.GapModel(_spec(names, [1.0, 0.5]))
    down = gk.GapModel(_spec(names, [1.0, 0.5], sign=-1.0))
    a = gk.gap_bias(up, _field(), (0, 0), ACTIONS, MASK, 1.0)
    b = gk.gap_bias(down, _field(), (0, 0), ACTIONS, MASK, 1.0)
    assert a.any() and np.allclose(a, -b) and a[-1] == 0 and abs(float(a[MASK].mean())) < 1e-5
    assert not gk.gap_bias(up, _field(), (0, 0), ACTIONS, MASK, 0.0).any()


def test_endgame_gate_makes_the_knob_inert_above_its_virus_limit():
    m = gk.GapModel(_spec(["holes"], [1.0], max_root_viruses=3))
    assert not gk.gap_bias(m, _field(), (0, 0), ACTIONS, MASK, 1.0).any()     # 4 viruses > 3
    m4 = gk.GapModel(_spec(["holes"], [1.0], max_root_viruses=4))
    assert gk.gap_bias(m4, _field(), (0, 0), ACTIONS, MASK, 1.0).any()


def test_virus_clear_counts_as_certain():
    m = gk.GapModel(_spec(["holes"], [0.0]))                  # a flat model: only the immediate rule matters
    pill = (0, 0)                                            # canonical 0 = NES red: onto the red floor viruses
    actions = np.array([1 * 128 + 14 * 8 + 3,                # vertical in column 3: completes the red floor row
                        1 * 128 + 14 * 8 + 6])               # vertical in column 6: no clear
    mask = np.array([True, True])
    b = gk.gap_bias(m, _field(), pill, actions, mask, 1.0, immediate="virus_clear")
    assert b[0] > 0 > b[1]


def test_bad_specs_are_refused():
    with pytest.raises(ValueError):
        gk.GapModel(dict(_spec(["holes"], [1.0]), schema="drmc-showy-knob-v1"))
    with pytest.raises(ValueError, match="unknown gap-knob features"):
        gk.GapModel(_spec(["no_such_feature"], [1.0]))


@pytest.mark.parametrize("key", GAP_KEYS)
def test_registered_gap_knobs_load_validate_and_stack(key):
    entry = knobs.parse(f"{key}:1.0")
    knobs.validate_knobs([entry])
    model = knobs.load_model(entry)
    assert isinstance(model, gk.GapModel) and model.spec["auc_heldout"] > 0.6
    one = knobs.total_bias([entry], _field(), (0, 0), ACTIONS, MASK)
    quad = knobs.parse("showy-quad@1:1.5")
    both = knobs.total_bias([quad, entry], _field(), (0, 0), ACTIONS, MASK)
    assert np.allclose(both, one + knobs.total_bias([quad], _field(), (0, 0), ACTIONS, MASK), atol=1e-5)
    assert knobs.suffix([quad, entry]) == f"+showy-quad@1:1.5+{key}:1"
    actor = SimpleNamespace(score=lambda obs, infos: None)
    assert knobs.apply(actor, [dict(entry, **{"lambda": 0.0})]) is actor


def test_bundled_gap_models_are_canonical_and_small():
    for key in GAP_KEYS:
        spec = json.loads((MODELS / knobs.REGISTRY[key].default_model).read_text())
        assert spec["schema"] == gk.SCHEMA and len(knobs.canonical(spec)) < knobs.MAX_MODEL_BYTES
