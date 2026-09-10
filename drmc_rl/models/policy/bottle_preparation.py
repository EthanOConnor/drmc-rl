"""Exact inference reuse for a G5 encoder trained without trunk conditioning.

Only the eight public bottle planes enter this cache. Preview, opponent,
history, motor context and candidate scoring remain fresh in the policy.
"""

from dataclasses import dataclass
import weakref

import torch


def _encoder_version(model):
    # Optimizers/load_state_dict update versions; device/dtype moves and tensor
    # replacement also change identity. Cache scope is this model instance.
    tensors = tuple(model.bottle.parameters()) + tuple(model.bottle.buffers())
    tensors += tuple(model.bottle_projection.parameters()) + tuple(model.bottle_projection.buffers())
    return tuple((id(t), t._version, t.data_ptr(), t.device, t.dtype) for t in tensors)


def _require_inference(model):
    if model.training or torch.is_grad_enabled():
        raise ValueError("bottle preparation is inference-only; use eval and no_grad/inference_mode")
    if model.conditioned_trunk:
        raise ValueError("conditioned encoder cannot prepare features without preview/history")


def _autocast_context(device):
    return torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type)


@dataclass(frozen=True)
class PreparedBottle:
    _model: weakref.ReferenceType
    _encoder_version: tuple
    _bottle: torch.Tensor
    _features: torch.Tensor
    _tensor_versions: tuple[int, int]
    _autocast: tuple

    def resolve(self, model, bottle):
        _require_inference(model)
        if self._model() is not model or self._encoder_version != _encoder_version(model):
            raise ValueError("prepared bottle belongs to a different or updated encoder")
        if self._tensor_versions != (self._bottle._version, self._features._version):
            raise ValueError("prepared bottle tensors were modified")
        if self._autocast != _autocast_context(bottle.device):
            raise ValueError("prepared bottle autocast context changed")
        if (bottle.device != self._bottle.device or bottle.dtype != self._bottle.dtype
                or bottle.shape[1:] != self._bottle.shape[1:]
                or self._bottle.shape[0] not in (1, bottle.shape[0])):
            raise ValueError("prepared bottle shape, device or dtype does not match")
        original = self._bottle.expand(bottle.shape[0], -1, -1, -1)
        if not torch.equal(original, bottle):
            raise ValueError("public bottle changed after preparation")
        return self._features.expand(bottle.shape[0], -1, -1, -1)


def prepare_bottle(model, bottle):
    _require_inference(model)
    if bottle.ndim != 4 or tuple(bottle.shape[1:]) != (8, 16, 8) or bottle.shape[0] < 1:
        raise ValueError("prepare a nonempty batch of eight public bottle planes")
    version = _encoder_version(model)
    # Ordinary detached tensors retain mutation counters even when the caller
    # uses inference_mode. No training graph or input alias is retained.
    with torch.inference_mode(False), torch.no_grad():
        original = bottle.detach().clone()
        condition = original.new_zeros((len(original), model.d_model))
        features = model.bottle_projection(model.bottle(original, condition))
    _require_inference(model)
    if version != _encoder_version(model):
        raise ValueError("encoder changed during bottle preparation")
    return PreparedBottle(weakref.ref(model), version, original, features,
                          (original._version, features._version), _autocast_context(bottle.device))
