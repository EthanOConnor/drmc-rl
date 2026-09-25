"""Showy-setup knob: opt-in wrapping, determinism and pool gating."""
import json
from pathlib import Path

import numpy as np
import pytest

from drmc_rl.pool import conditions, store
from drmc_rl.style import showy_knob as sk

MODEL = json.loads((Path(sk.__file__).parent / "models" / "showy_t2k4_v1.json").read_text())


def _field():
    f = np.full(128, 0xFF, np.uint8)
    f[120:124] = (0xD1, 0xD1, 0xD1, 0xFF)   # three red viruses on the floor
    f[112] = 0xD0
    return f


def test_features_and_bias_are_deterministic():
    f = _field()
    actions = np.array([0 * 128 + 115, 1 * 128 + 99, 0 * 128 + 96, -1])
    mask = np.array([True, True, True, False])
    m = sk.ShowyModel.load(MODEL)
    a = sk.showy_bias(m, f, (0, 0), actions, mask, 1.5)
    b = sk.showy_bias(m, f, (0, 0), actions, mask, 1.5)
    assert np.array_equal(a, b) and a[3] == 0 and np.isfinite(a).all()
    assert not sk.showy_bias(m, f, (0, 0), actions, mask, 0.0).any()


def test_capability_and_requirements():
    repo = Path(__file__).resolve().parents[1]
    assert conditions.SHOWY_CAPABILITY in conditions.runtime_capabilities(repo)
    plain = dict(loader="plain", settings={})
    knob = dict(loader="plain", settings=dict(showy_lambda=1.5, showy_model=MODEL))
    assert conditions.SHOWY_CAPABILITY not in store.entrant_requirements(plain)
    assert conditions.SHOWY_CAPABILITY in store.entrant_requirements(knob)


def _record(**settings):
    return dict(id="knob-champ-l1.5", status="active", loader="plain", era="style",
                checkpoint=dict(sha256="0" * 64), settings=settings, requires=[conditions.SHOWY_CAPABILITY])


def test_entrant_validation():
    store.validate_entrant(_record(showy_lambda=1.5, showy_model=MODEL))
    with pytest.raises(ValueError):
        store.validate_entrant(_record(showy_lambda=1.5, showy_model="/local/path.json"))
    with pytest.raises(ValueError):
        store.validate_entrant(_record(showy_lambda=0, showy_model=MODEL))
    with pytest.raises(ValueError):
        store.validate_entrant(_record(showy_lambda=1.5, showy_model=MODEL, showy_bogus=1))
    missing = _record(showy_lambda=1.5, showy_model=MODEL)
    missing["requires"] = []
    with pytest.raises(ValueError):
        store.validate_entrant(missing)
