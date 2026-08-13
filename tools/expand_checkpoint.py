"""Checkpoint surgery: expand a candidate-policy checkpoint to new input dims.

Zero-initializes the weight slices for newly added board channels (e.g. the
opponent-bottle planes of ``state_repr=bitplane_bottle_conn_mask_vs``) and
auxiliary scalars (``aux_spec=v1_vs``), so the expanded net's outputs are
exactly identical to the old net at init regardless of what the new inputs
contain. Applies to both ``state_dict`` and ``ema_state_dict``; the optimizer
state is dropped (shapes no longer match — start the new run with
``resume_optimizer: false``).

Only the CNN board encoder (`_BoardCNN`) is supported: its stem conv input is
``[board_channels | coord_y | coord_x]``, so expansion copies the old board
slices, zero-fills the new channels, and moves the coord slices to the end.

Example:
    python -m tools.expand_checkpoint \
        --in runs/best_agents/smdp_ppo_step535164979.pt.gz \
        --out runs/best_agents/smdp_ppo_step535164979_vsobs.pt.gz \
        --config drmc_rl/training/configs/vs_selfplay.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from drmc_rl.training.algo.ppo_smdp import _AUX_DIM_BY_SPEC
from drmc_rl.training.utils.checkpoint_io import load_checkpoint, save_checkpoint

_STEM_KEY = "board_trunk.stem.0.weight"
_AUX_KEY = "aux_encoder.0.weight"
_COORD_CH = 2  # CoordConv y/x channels appended after the board planes


def expand_state_dict(
    sd: Dict[str, torch.Tensor],
    *,
    new_board_channels: int,
    new_aux_dim: int,
) -> Dict[str, torch.Tensor]:
    """Return a copy of ``sd`` expanded to the new input dims (zero-init slices)."""

    out = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in sd.items()}

    stem = out.get(_STEM_KEY)
    if stem is None:
        raise KeyError(
            f"{_STEM_KEY} not found; only candidate_board_encoder=cnn checkpoints are supported"
        )
    old_board = int(stem.shape[1]) - _COORD_CH
    if new_board_channels < old_board:
        raise ValueError(f"new board_channels {new_board_channels} < old {old_board}")
    if new_board_channels > old_board:
        new_stem = stem.new_zeros((stem.shape[0], new_board_channels + _COORD_CH, *stem.shape[2:]))
        new_stem[:, :old_board] = stem[:, :old_board]
        new_stem[:, new_board_channels:] = stem[:, old_board:]  # coord channels
        out[_STEM_KEY] = new_stem

    aux_w = out.get(_AUX_KEY)
    if new_aux_dim > 0:
        if aux_w is None:
            raise KeyError(f"{_AUX_KEY} not found but new aux_dim={new_aux_dim} requested")
        old_aux = int(aux_w.shape[1])
        if new_aux_dim < old_aux:
            raise ValueError(f"new aux_dim {new_aux_dim} < old {old_aux}")
        if new_aux_dim > old_aux:
            new_aux_w = aux_w.new_zeros((aux_w.shape[0], new_aux_dim))
            new_aux_w[:, :old_aux] = aux_w
            out[_AUX_KEY] = new_aux_w

    return out


def expand_checkpoint_payload(
    payload: Dict[str, Any],
    *,
    new_board_channels: int,
    new_aux_dim: int,
    new_smdp_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Expand a trainer checkpoint payload in place (returns it for chaining)."""

    for key in ("state_dict", "ema_state_dict"):
        sd = payload.get(key)
        if sd is not None:
            payload[key] = expand_state_dict(
                sd, new_board_channels=new_board_channels, new_aux_dim=new_aux_dim
            )
    # Optimizer moment shapes no longer match the expanded weights.
    payload.pop("optimizer", None)
    # Keep the embedded cfg loadable by the opponent pool / eval tools.
    if new_smdp_cfg is not None:
        cfg = dict(payload.get("cfg") or {})
        cfg["smdp_ppo"] = dict(new_smdp_cfg)
        payload["cfg"] = cfg
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, help="old checkpoint (.pt / .pt.gz)")
    parser.add_argument("--out", dest="dst", required=True, help="expanded checkpoint path")
    parser.add_argument(
        "--config",
        required=True,
        help="run config yaml whose smdp_ppo section defines the new dims "
        "(candidate_board_channels, aux_spec)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    sp = dict(cfg.get("smdp_ppo") or {})
    new_board_channels = int(sp.get("candidate_board_channels", 0))
    if new_board_channels <= 0:
        raise SystemExit("--config must set smdp_ppo.candidate_board_channels explicitly")
    new_aux_dim = int(_AUX_DIM_BY_SPEC.get(str(sp.get("aux_spec", "none")).strip().lower(), 0))

    payload = load_checkpoint(Path(args.src), map_location="cpu")
    expand_checkpoint_payload(
        payload,
        new_board_channels=new_board_channels,
        new_aux_dim=new_aux_dim,
        new_smdp_cfg=sp,
    )
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(payload, dst)
    print(
        f"wrote {dst} (board_channels={new_board_channels}, aux_dim={new_aux_dim}, "
        f"step={payload.get('step')})"
    )


if __name__ == "__main__":
    main()
