"""The optional native knob-feature path equals the numpy reference bit for bit.

Runs on deterministic synthetic decisions; set ``DRMC_KNOB_DECISIONS`` to a
recorded-decisions file (``python -m tools.showy_knob.bench record``) to run the
same checks on real games. Builds the library into a temp dir when it is absent
and a C compiler exists; otherwise the native checks skip (the numpy fallback
is still checked).
"""
import os
import shutil

import numpy as np
import pytest

from drmc_rl.style import knobs, native
from drmc_rl.style import showy_knob as sk
from tools.showy_knob.bench import SPECS, load_decisions, synthetic_decisions


@pytest.fixture(scope="module")
def rows():
    path = os.environ.get("DRMC_KNOB_DECISIONS")
    return load_decisions(path) if path else synthetic_decisions(2000, seed=7)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    old = {k: os.environ.get(k) for k in ("DRMC_KNOB_NATIVE", "DRMC_KNOB_NATIVE_LIB")}
    os.environ.pop("DRMC_KNOB_NATIVE", None)
    native.reset()
    if not native.available():
        if not shutil.which(os.environ.get("CC", "cc")):
            pytest.skip("no native library and no C compiler")
        path = tmp_path_factory.mktemp("knobnative") / native.library_path().name
        os.environ["DRMC_KNOB_NATIVE_LIB"] = str(path)
        native.build(path)
    assert native.available()
    yield native.load()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    native.reset()


@pytest.fixture(autouse=True)
def _restore_mode(monkeypatch):
    monkeypatch.setenv("DRMC_KNOB_NATIVE", os.environ.get("DRMC_KNOB_NATIVE", "1"))
    yield
    native.reset()


def _mode(on):
    os.environ["DRMC_KNOB_NATIVE"] = "1" if on else "0"
    native.reset()


def test_feature_names_are_static():
    assert len(sk.FEATURE_NAMES) == 58
    x = sk.board_features(np.full((1, 128), 0xFF, np.uint8))
    assert x.shape == (1, 58)


def test_native_features_equal_reference(lib, rows):
    _mode(True)
    for f, p, a, m in rows:
        legal, colors = np.flatnonzero(m), (int(p[0]), int(p[1]))
        after, x, score, lines = native.decision_features(f, colors, a[legal], True)
        ref_after = sk.reference_afterstates(np.asarray(f, np.uint8).reshape(128), colors, a[legal])
        assert after.tobytes() == ref_after.tobytes()
        ref_x = np.concatenate([sk.board_features(ref_after), sk.trigger_features(ref_after)], axis=1)
        assert x.dtype == ref_x.dtype and x.tobytes() == ref_x.tobytes()
        s0, l0 = sk.immediate_clears(f, colors, a[legal])
        assert score.tobytes() == s0.tobytes() and lines.tobytes() == l0.tobytes()


@pytest.mark.parametrize("spec", SPECS)
def test_native_bias_is_byte_identical(lib, rows, spec):
    entries = [knobs.parse(s) for s in spec.split(",")]
    models = [knobs.load_model(e) for e in entries]
    out = {}
    for on in (False, True):
        _mode(on)
        out[on] = [knobs.total_bias(entries, f, p, a, m, models=models) for f, p, a, m in rows[:150]]
    assert all(x.tobytes() == y.tobytes() for x, y in zip(out[False], out[True]))


def test_stacked_knobs_equal_the_sum_of_single_knobs(rows):
    """The shared per-decision cache changes nothing: a stack is the float32 sum of its terms."""
    _mode(False)
    entries = [knobs.parse("showy-t2@1:1.5"), knobs.parse("showy-hcombo@1:0.5")]
    models = [knobs.load_model(e) for e in entries]
    for f, p, a, m in rows[:40]:
        expect = np.zeros(len(a), np.float32)
        for e, mdl in zip(entries, models):
            expect += knobs.total_bias([e], f, p, a, m, models=[mdl])
        assert knobs.total_bias(entries, f, p, a, m, models=models).tobytes() == expect.tobytes()


def test_bad_placement_raises_like_the_reference(lib):
    f = np.full(128, 0xFF, np.uint8)
    f[120] = 0xD1
    mask = np.array([True])
    for on in (False, True):
        _mode(on)
        with pytest.raises(ValueError):
            knobs.total_bias([knobs.parse("showy-t2@1:1")], f, (0, 0), np.array([120]), mask)
