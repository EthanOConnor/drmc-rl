import base64
import json
import struct

import pytest

from drmc_rl.human.professor_corpus import corpus_report, decode_pptape


def _tape(runs):
    frames = sum(row[0] for row in runs)
    header = json.dumps({"schema": "pptape1", "frames": frames}).encode()
    data = b"PPTAPE1\0" + struct.pack("<I", len(header)) + header + struct.pack("<I", 0)
    data += struct.pack("<I", len(runs))
    for count, p1, p2 in runs:
        data += struct.pack("<IBB", count, p1, p2)
    return data


def test_decode_rejects_frame_mismatch():
    tape = bytearray(_tape([(10, 0, 0)]))
    header_start = 12
    header_end = header_start + struct.unpack("<I", tape[8:12])[0]
    header = json.loads(tape[header_start:header_end])
    header["frames"] = 11
    replacement = json.dumps(header).encode()
    assert len(replacement) == header_end - header_start
    tape[header_start:header_end] = replacement
    with pytest.raises(ValueError, match="frame count"):
        decode_pptape(bytes(tape))


def test_report_exposes_identity_cadence_and_ai_calibration(tmp_path):
    tape = _tape([(60, 0, 0), (1, 0x80, 0x01), (59, 0, 0)])
    encoded = base64.urlsafe_b64encode(tape).rstrip(b"=").decode()
    artifact = {
        "payload": {
            "mode": "versus",
            "tape": encoded,
            "summary": {
                "p1_wins": 3,
                "p2_wins": 1,
                "winner_side": 1,
                "player_two": {
                    "kind": "drmc-human-backend-v1",
                    "target_rating": 1600,
                    "timing_scale_milli": 1000,
                },
            },
        }
    }
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({
        "schema": "ppcorpus1",
        "run_id": "abc",
        "account": {"handle": "human"},
        "artifact": artifact,
    }) + "\n")
    report = corpus_report(path)
    assert report["runs"] == 1
    assert report["players"]["human"]["raw_actions_per_minute"] == pytest.approx(30.0494)
    assert report["calibration_matches"][0] == {
        **report["calibration_matches"][0],
        "human": "human",
        "target_rating": 1600.0,
        "human_wins": 3,
        "model_wins": 1,
    }
