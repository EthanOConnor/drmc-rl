"""Adapters for libretro action spaces (Discrete or MultiBinary)."""
from __future__ import annotations

from typing import List

from envs.backends.base import NES_BUTTONS


def discrete10_to_buttons(a: int, held: dict) -> List[int]:
    """Map our 10-action discrete index to a NES MultiBinary[8] button vector.

    held is a dict tracking latched holds for LEFT/RIGHT/DOWN: {'LEFT':bool, 'RIGHT':bool, 'DOWN':bool}
    We return an 8-length list of 0/1 in NES_BUTTONS order.
    """
    b = {k: 0 for k in NES_BUTTONS}
    # apply holds first
    if held.get('LEFT'):  b['LEFT'] = 1
    if held.get('RIGHT'): b['RIGHT'] = 1
    if held.get('DOWN'):  b['DOWN'] = 1

    if a == 1:
        b['LEFT'] = 1;  held.update({'LEFT': True, 'RIGHT': False})
    elif a == 2:
        b['RIGHT'] = 1; held.update({'RIGHT': True, 'LEFT': False})
    elif a == 3:
        b['DOWN'] = 1;  held.update({'DOWN': True})
    elif a == 4:
        b['A'] = 1   # rotate_A
    elif a == 5:
        b['B'] = 1   # rotate_B
    elif a == 6:
        held.update({'LEFT': True, 'RIGHT': False})
    elif a == 7:
        held.update({'RIGHT': True, 'LEFT': False})
    elif a == 8:
        held.update({'DOWN': True})
    elif a == 9:
        b['A'] = 1; b['B'] = 1  # both_rot
    else:
        pass  # noop

    # Convert dict to array in NES_BUTTONS order
    return [b[k] for k in NES_BUTTONS]
