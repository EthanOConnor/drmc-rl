"""Decision-time knobs: registry, stacking, identity at lambda 0, and pool gating."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.pool import conditions, store
from drmc_rl.style import knobs
from drmc_rl.style import showy_knob as sk

MODEL = json.loads((Path(sk.__file__).parent / "models" / "showy_t2k4_v1.json").read_text())
HC = json.loads((Path(sk.__file__).parent / "models" / "showy_hc_k4_v1.json").read_text())


def _field():
    f = np.full(128, 0xFF, np.uint8)
    f[120:124] = (0xD1, 0xD1, 0xD1, 0xFF)   # three red viruses on the floor
    f[112] = 0xD0
    return f


ACTIONS = np.array([0 * 128 + 115, 1 * 128 + 99, 0 * 128 + 96, 0 * 128 + 100, -1])
MASK = np.array([True, True, True, True, False])


def knob(ident, lam, model=MODEL, version=1):
    return {"id": ident, "version": version, "lambda": lam, "model": model}


def test_features_and_bias_are_deterministic():
    m = sk.ShowyModel.load(MODEL)
    a = sk.showy_bias(m, _field(), (0, 0), ACTIONS, MASK, 1.5)
    assert np.array_equal(a, sk.showy_bias(m, _field(), (0, 0), ACTIONS, MASK, 1.5)) and a[4] == 0
    assert not sk.showy_bias(m, _field(), (0, 0), ACTIONS, MASK, 0.0).any()


def test_every_knob_at_lambda_zero_is_the_unchanged_actor():
    actor = SimpleNamespace(score=lambda obs, infos: None)
    for key, entry in knobs.REGISTRY.items():
        zero = [knob(entry.id, 0.0, model=knobs.default_model_ref(key))]
        assert knobs.apply(actor, zero) is actor
        assert not knobs.total_bias(zero, _field(), (0, 0), ACTIONS, MASK).any()
    assert knobs.apply(actor, []) is actor and knobs.apply(actor, None) is actor


def test_stacked_knobs_are_the_sum_of_their_biases():
    one = [knob("showy-t2", 1.5)]
    two = [knob("showy-hcombo", 0.7, model=HC)]
    both = one + two
    t = knobs.total_bias(both, _field(), (0, 0), ACTIONS, MASK)
    s = knobs.total_bias(one, _field(), (0, 0), ACTIONS, MASK) + knobs.total_bias(two, _field(), (0, 0), ACTIONS, MASK)
    assert np.allclose(t, s, atol=1e-6) and t[4] == 0 and t.any()
    assert abs(float(t[MASK].mean())) < 1e-5                  # each bias is centered over legal candidates


def test_unknown_knobs_versions_keys_and_paths_are_refused():
    with pytest.raises(ValueError, match="unknown knob"):
        knobs.validate_knobs([knob("showy-bogus", 1.0)])
    with pytest.raises(ValueError, match="unknown knob"):
        knobs.validate_knobs([knob("showy-t2", 1.0, version=2)])          # a version bump is another knob
    with pytest.raises(ValueError):
        knobs.validate_knobs([dict(knob("showy-t2", 1.0), extra=1)])
    with pytest.raises(ValueError):
        knobs.validate_knobs([knob("showy-t2", 1.0, model="/local/path.json")])
    with pytest.raises(ValueError):
        conditions.validate_knob_settings(dict(showy_lambda=1.5, showy_model=MODEL))   # old form dropped


def test_capabilities_requirements_and_identity():
    repo = Path(__file__).resolve().parents[1]
    caps = conditions.runtime_capabilities(repo)
    assert {"knob:showy-t2@1", "knob:showy-hcombo@1"} <= caps
    entries = [knob("showy-t2", 1.5), knob("showy-hcombo", 0.5, model=HC), knob("showy-hcombo", 0.0, model=HC, version=1)][:2]
    record = dict(loader="plain", settings=dict(knobs=entries))
    assert store.entrant_requirements(record) >= {"knob:showy-t2@1", "knob:showy-hcombo@1"}
    assert knobs.suffix(entries) == "+showy-t2@1:1.5+showy-hcombo@1:0.5"
    assert conditions.check_name("bigclear-champ-f100M" + knobs.suffix(entries))
    assert knobs.parse("showy-t2@1:1.25")["model"] == knobs.default_model_ref("showy-t2@1")


def _record(**settings):
    return dict(id="champ+showy-t2@1:1.5", status="active", loader="plain", era="style",
                checkpoint=dict(sha256="0" * 64), settings=settings, requires=["knob:showy-t2@1"])


def test_entrant_validation():
    store.validate_entrant(_record(knobs=[knob("showy-t2", 1.5)]))
    missing = _record(knobs=[knob("showy-t2", 1.5)])
    missing["requires"] = []
    with pytest.raises(ValueError):
        store.validate_entrant(missing)


QUAD = json.loads((Path(sk.__file__).parent / "models" / "showy_quad_k4_v1.json").read_text())


def _quad_field():
    """A horizontal red/yellow pill at row 12, cols 3-4 completes four lines at once (a quad)."""
    f = np.full(128, 0xFF, np.uint8)
    R, Y, B = 0x01, 0x00, 0x02
    for r in range(13, 16):
        f[r * 8 + 3] = 0xD0 | R
        f[r * 8 + 4] = 0xD0 | Y
        for c in (0, 1, 2, 5, 6, 7):
            f[r * 8 + c] = 0xD0 | B
    for c in (0, 1, 2):
        f[12 * 8 + c] = 0xD0 | R
    for c in (5, 6, 7):
        f[12 * 8 + c] = 0xD0 | Y
    return f


def test_quad_override_scores_the_placements_own_attack():
    f = _quad_field()
    actions = np.array([99, 1 * 128 + 83, 0 * 128 + 88, -1])     # the quad, and two quiet placements
    mask = np.array([True, True, True, False])
    score, lines = sk.immediate_clears(f, (0, 1), actions[:3])
    assert lines.tolist() == [4, 0, 0]
    m = sk.ShowyModel.load(QUAD)
    b = sk.quad_bias(m, f, (0, 1), actions, mask, 1.0)
    assert b[0] == b[mask].max() and b[0] > b[1] + 3 and b[3] == 0
    assert abs(float(b[mask].mean())) < 1e-5


def test_quad_stacks_with_showy_t2_and_is_inert_at_zero():
    ref = knobs.default_model_ref("showy-quad@1")
    both = [knob("showy-t2", 1.0), knob("showy-quad", 1.5, model=ref)]
    t = knobs.total_bias(both, _field(), (0, 0), ACTIONS, MASK)
    s = (knobs.total_bias(both[:1], _field(), (0, 0), ACTIONS, MASK)
         + knobs.total_bias(both[1:], _field(), (0, 0), ACTIONS, MASK))
    assert np.allclose(t, s, atol=1e-6) and t[4] == 0
    zero = [knob("showy-t2", 1.0), knob("showy-quad", 0.0, model=ref)]
    assert np.array_equal(knobs.total_bias(zero, _field(), (0, 0), ACTIONS, MASK),
                          knobs.total_bias(both[:1], _field(), (0, 0), ACTIONS, MASK))
    assert "knob:showy-quad@1" in conditions.runtime_capabilities(Path(__file__).resolve().parents[1])
    assert knobs.suffix(both) == "+showy-t2@1:1+showy-quad@1:1.5"


def test_waste_penalty_lowers_only_over_cap_attacks():
    f = _quad_field()
    actions = np.array([99, 1 * 128 + 83]); mask = np.array([True, True])
    plain = sk.quad_bias(sk.ShowyModel.load(QUAD), f, (0, 1), actions, mask, 1.0)
    penal = sk.quad_bias(sk.ShowyModel.load(dict(QUAD, waste_penalty=2.0)), f, (0, 1), actions, mask, 1.0)
    assert np.allclose(plain, penal)                     # exactly 4 lines: nothing wasted


def test_older_models_ignore_appended_trigger_columns():
    boards = np.stack([_field(), _quad_field()])
    t = sk.trigger_features(boards)
    assert t.shape == (2, len(sk.TRIGGER_NAMES)) and len(sk.TRIGGER_NAMES) == 12
    m = sk.ShowyModel.load(MODEL)
    assert np.isfinite(m.logit(boards)).all()
