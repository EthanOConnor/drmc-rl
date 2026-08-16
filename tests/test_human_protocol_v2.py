import pytest

from drmc_rl.game.pair_state import DecisionBoundary, PublicPairState, VisibleSideState
from drmc_rl.human.protocol_v2 import DecisionRequestV2, PROTOCOL_SCHEMA


def state():
    side = VisibleSideState(bytes([0xFF] * 128), (0, 1), (2, 0), None)
    return PublicPairState(10, 0, (side, side), DecisionBoundary.P1).to_dict()


def test_protocol_requires_product_specific_controls() -> None:
    request = DecisionRequestV2.from_mapping(
        {
            "schema": PROTOCOL_SCHEMA,
            "type": "decide",
            "request_id": 1,
            "frame_id": 10,
            "deadline_ms": 90,
            "product": "trainer",
            "controls": {"target_rating": 1700, "style": [0.2, -0.1]},
            "state": state(),
        }
    )
    assert request.controls.target_rating == 1700
    with pytest.raises(ValueError, match="execution_profile"):
        DecisionRequestV2.from_mapping(
            {
                "schema": PROTOCOL_SCHEMA,
                "request_id": 2,
                "frame_id": 10,
                "deadline_ms": 90,
                "product": "human_rate",
                "controls": {},
                "state": state(),
            }
        )
