import numpy as np
import torch

from drmc_rl.models.policy.effect_tokens import EFFECT_TOKEN_DIM, build_effect_tokens


def test_effect_tokens_capture_virus_and_height_change() -> None:
    root = np.full((1, 128), 0xFF, dtype=np.uint8)
    root[0, 15 * 8] = 0xD0
    after = np.repeat(root[:, None], 2, axis=1)
    after[0, 0, 15 * 8] = 0xFF
    after[0, 1, 0] = 0x80
    mask = np.array([[True, False]])
    tokens = build_effect_tokens(root, after, mask, viruses_cleared=np.array([[1, 0]]))
    assert tokens.shape == (1, 2, EFFECT_TOKEN_DIM)
    assert torch.count_nonzero(tokens[0, 0]) > 0
    assert torch.count_nonzero(tokens[0, 1]) == 0
