from __future__ import annotations

import numpy as np

from training.diagnostics.video import VideoWriter


def test_video_writer(tmp_path) -> None:
    writer = VideoWriter(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(3):
        writer.add_frame("rollout", frame)
    path = writer.finalize("rollout", 12)
    assert path.exists()
    assert path.stat().st_size > 0
