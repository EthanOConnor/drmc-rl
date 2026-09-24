from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.game.afterstate import (
    FACT_NAMES,
    _FACT_SCALE,
    afterstate_batch,
    planes_to_fields,
    resolve_placement,
)
from drmc_rl.game.observation import board_bytes_to_semantic_planes

torch = pytest.importorskip("torch")


def _random_fields(rng, count):
    tiles = [0xFF] + [kind | color for kind in (0x40, 0x50, 0x60, 0x70, 0x80, 0xD0) for color in range(3)]
    return rng.choice(np.asarray(tiles, np.uint8), size=(count, 128), p=None)


def test_semantic_planes_round_trip_every_tile_kind():
    rng = np.random.default_rng(0)
    fields = _random_fields(rng, 64)
    planes = np.stack([board_bytes_to_semantic_planes(f) for f in fields])
    assert np.array_equal(planes_to_fields(planes), fields)


def _field(cells):
    field = np.full(128, 0xFF, np.uint8)
    for (row, col), tile in cells.items():
        field[row * 8 + col] = tile
    return field


def test_two_line_combo_sends_two_garbage_with_line_colors():
    # Column 0 holds red, column 1 yellow (NES low nibble Y=0, R=1, B=2).
    field = _field({
        (13, 0): 0x81, (14, 0): 0x81, (15, 0): 0xD1,
        (13, 1): 0x80, (14, 1): 0x80, (15, 1): 0xD0,
    })
    # Canonical R=0, Y=1: anchor red at (12,0), yellow to its right.
    after, raw = resolve_placement(field, (0, 1), 0 * 128 + 12 * 8 + 0)
    facts = dict(zip(FACT_NAMES, raw))
    assert set(after) == {0xFF}
    assert facts["lines"] == 2 and facts["rounds"] == 1 and facts["tiles_cleared"] == 8
    assert facts["viruses_cleared"] == 2 and facts["viruses_after"] == 0 and facts["win"] == 1
    assert facts["garbage_sent"] == 2
    assert (facts["garbage_red"], facts["garbage_yellow"], facts["garbage_blue"]) == (1, 1, 0)


def test_single_line_sends_no_garbage():
    field = _field({(13, 0): 0x81, (14, 0): 0x81, (15, 0): 0xD1, (15, 1): 0xD2})
    after, raw = resolve_placement(field, (0, 0), 3 * 128 + 12 * 8 + 0)  # red at 12, red above
    facts = dict(zip(FACT_NAMES, raw))
    assert facts["lines"] == 1 and facts["garbage_sent"] == 0 and facts["tiles_cleared"] == 5
    assert facts["viruses_cleared"] == 1 and facts["viruses_after"] == 1 and facts["win"] == 0
    board = np.frombuffer(after, np.uint8)
    assert board[15 * 8 + 1] == 0xD2 and (board != 0xFF).sum() == 1


def test_spawn_blocked_is_reported():
    field = _field({(r, 3): 0x80 | (r % 3) for r in range(2, 16)})
    _after, raw = resolve_placement(field, (2, 2), 1 * 128 + 0 * 8 + 3)
    assert dict(zip(FACT_NAMES, raw))["spawn_blocked"] == 1


def test_batch_matches_single_placements_and_pads_with_root():
    rng = np.random.default_rng(1)
    field = _field({(15, c): 0xD0 | (c % 3) for c in range(8)})
    planes = board_bytes_to_semantic_planes(field)[None]
    actions = np.asarray([[0 * 128 + 14 * 8 + 0, 1 * 128 + 13 * 8 + 5, 0 * 128 + 14 * 8 + 0, -1]])
    mask = np.asarray([[True, True, True, False]])
    tiles, facts = afterstate_batch(planes, np.asarray([[0, 2]]), actions, mask)
    for k in range(3):
        after, raw = resolve_placement(field, (0, 2), int(actions[0, k]))
        assert np.array_equal(tiles[0, k], np.frombuffer(after, np.uint8))
        assert np.allclose(facts[0, k], raw / _FACT_SCALE)
    assert np.array_equal(tiles[0, 3], field) and not facts[0, 3].any()
    del rng


def test_native_engine_parity_on_virus_bottles():
    try:
        from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator

        sim = NativeAfterstateSimulator(num_envs=64)
    except Exception as error:  # pragma: no cover - native library absent
        pytest.skip(f"native pool unavailable: {error}")
    from drmc_rl.game.afterstate import _CANONICAL_TO_NES
    from drmc_rl.seedlab.rng import place_viruses

    rng = np.random.default_rng(3)
    fields, pills, actions, counts = [], [], [], []
    for level in (4, 10, 16, 20):
        for seed in range(6):
            board, _r0, _r1 = place_viruses(level, int(rng.integers(1, 255)), int(rng.integers(1, 255)))
            field = np.frombuffer(bytes(board), np.uint8).copy()
            poses = []
            for action in rng.permutation(512):
                o, cell = divmod(int(action), 128)
                r, c = divmod(cell, 8)
                dr, dc = ((0, 1), (1, 0), (0, -1), (-1, 0))[o]
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < 16 and 0 <= c2 < 8) or field[cell] != 0xFF or field[r2 * 8 + c2] != 0xFF:
                    continue
                below = [rr * 8 + cc for rr, cc in ((r + 1, c), (r2 + 1, c2)) if rr < 16]
                if len(below) == 2 and all(field[b] == 0xFF for b in below):
                    continue  # locks only where at least one half rests
                top = min(r, r2)
                if any(field[rr * 8 + cc] != 0xFF for cc in {c, c2} for rr in range(top)):
                    continue  # open sky above, so the injected script can reach it
                poses.append(int(action))
                if len(poses) == 24:
                    break
            fields.append(field)
            pills.append(rng.integers(0, 3, 2))
            actions.append(poses)
            counts.append(len(poses))
    width = max(counts)
    A = np.full((len(fields), width), -1, np.int64)
    M = np.zeros((len(fields), width), bool)
    for i, poses in enumerate(actions):
        A[i, : len(poses)] = poses
        M[i, : len(poses)] = True
    planes = np.stack([board_bytes_to_semantic_planes(f) for f in fields])
    pills = np.stack(pills)
    tiles, facts = afterstate_batch(planes, pills, A, M)
    try:
        out = sim.simulate_packed(
            fields=np.stack(fields), pills=_CANONICAL_TO_NES[pills], previews=_CANONICAL_TO_NES[pills],
            candidate_actions=A, candidate_costs=np.zeros_like(A, dtype=np.uint16),
            candidate_count=np.asarray(counts), speed=np.full(len(fields), 2), speed_ups=np.zeros(len(fields), int),
        )
    finally:
        sim.close()
    ongoing = out.terminal_reason == 0
    assert ongoing.mean() > 0.9
    assert np.array_equal(tiles[M][ongoing], out.fields[ongoing])
    raw = facts[M] * _FACT_SCALE
    assert np.array_equal(np.rint(raw[:, 0]), out.viruses_cleared)
    assert np.array_equal(np.rint(raw[:, 1] - raw[:, 0]), out.nonviruses_cleared)


def _small_net():
    from drmc_rl.models.policy.afterstate_core import afterstate_core_config, build_afterstate_core

    cfg = afterstate_core_config(encoder_blocks=1, candidate_d_model=64, pill_embed_dim=16,
                                 candidate_hidden_dim=64, candidate_cross_layers=1,
                                 candidate_interaction_layers=1, candidate_transformer_heads=4)
    torch.manual_seed(0)
    return cfg, build_afterstate_core(cfg["smdp_ppo"], 20, 733).eval()


def _inputs(rng, batch=3, width=12):
    fields = np.stack([_field({(15, c): 0xD0 | (c % 3) for c in range(8)}) for _ in range(batch)])
    for f in fields:
        f[rng.integers(64, 120, 6)] = 0x81
    planes = np.stack([board_bytes_to_semantic_planes(f) for f in fields])
    obs = np.concatenate((planes, planes[:, ::-1], np.zeros((batch, 4, 16, 8), np.float32)), axis=1)
    actions = np.full((batch, width), -1, np.int64)
    mask = np.zeros((batch, width), bool)
    for b in range(batch):
        free = [a for a in range(0, 128) if fields[b][a] == 0xFF and a % 8 < 7 and fields[b][a + 1] == 0xFF]
        n = width - b * 2
        actions[b, :n] = free[-n:]
        mask[b, :n] = True
    aux = np.zeros((batch, 733), np.float32)
    aux[:, 0] = 1.0
    return tuple(torch.from_numpy(x) for x in (
        obs.astype(np.float32), rng.integers(0, 3, (batch, 2)), rng.integers(0, 3, (batch, 2)),
        actions, rng.random((batch, width)).astype(np.float32) * 40, mask)), torch.from_numpy(aux)


def test_model_packed_and_dense_paths_agree_and_rows_are_independent(monkeypatch):
    _cfg, net = _small_net()
    inputs, aux = _inputs(np.random.default_rng(5))
    with torch.inference_mode():
        logits, value = net(*inputs, aux=aux)
        single, single_value = net(*(t[1:2] for t in inputs), aux=aux[1:2])
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        dense, _ = net(*inputs, aux=aux)
    mask = inputs[5]
    assert torch.isfinite(logits[mask]).all() and (logits[~mask] < -1e8).all()
    assert torch.allclose(dense[mask], logits[mask], atol=1e-5)
    n = int(mask[1].sum())
    assert torch.allclose(single[0, :n], logits[1, :n], atol=1e-4)
    assert torch.allclose(single_value, value[1:2], atol=1e-5)


def test_model_uses_afterstates_and_loads_through_plain_policy(tmp_path):
    from tools.vs_head_to_head import PlainPolicy

    cfg, net = _small_net()
    inputs, aux = _inputs(np.random.default_rng(6))
    tiles, facts = net.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
    cached = net.exact_afterstates(inputs[0], inputs[1], inputs[3], inputs[5])
    assert torch.equal(cached[0], tiles) and torch.equal(cached[1], facts)
    with torch.inference_mode():
        reference, _ = net(*inputs, aux=aux)
        explicit, _ = net(*inputs, aux=aux, afterstate=(tiles, facts))
        changed, _ = net(*inputs, aux=aux, afterstate=(tiles, torch.zeros_like(facts)))
    assert torch.equal(reference, explicit)
    assert not torch.allclose(reference[inputs[5]], changed[inputs[5]])
    path = tmp_path / "student.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), path)
    policy = PlainPolicy(path, "cpu", public_only=True)
    with torch.inference_mode():
        loaded, _ = policy.net(*inputs, aux=aux)
    assert torch.equal(loaded, reference)


def test_controller_core_trainer_accepts_the_afterstate_core(tmp_path):
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy

    cfg, net = _small_net()
    path = tmp_path / "student.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), path)
    actor = ControllerCorePolicy(path, "cpu", training=True)
    inputs, aux = _inputs(np.random.default_rng(7))
    records = []
    for b in range(inputs[0].shape[0]):
        n = int(inputs[5][b].sum())
        records.append(dict(
            observation=inputs[0][b].numpy().astype(np.uint8), pill=inputs[1][b].numpy().astype(np.int8),
            preview=inputs[2][b].numpy().astype(np.int8), actions=inputs[3][b, :n].numpy().astype(np.int16),
            costs=inputs[4][b, :n].numpy().astype(np.uint16), mask=np.ones(n, bool),
            public_context=aux[b].numpy(), base_logits=np.zeros(n, np.float32), slot=0))
    features, data = actor.training_batch(records)
    logits, value = actor.training_forward(features)
    assert logits.shape[0] == len(records) and value.shape == (len(records),)
    (logits.log_softmax(-1)[:, 0].sum() + value.sum()).backward()
    assert net is not actor.net and actor.net.bottle.stem.weight.grad is not None
