from drmc_rl.experiments.earliest_lock import EarliestLockExperiment, TimingProbe


class Backend:
    def evaluate(self, probe, lock_frame):
        return {
            "side_frames": [lock_frame, 10],
            "need_action": [True, True],
            "outcome": [0, 0],
            "terminated": [False],
            "truncated": [False],
            "board_bytes": [[lock_frame % 2], [0]],
        }, float(lock_frame == 11)


class ClockOnlyBackend:
    def evaluate(self, probe, lock_frame):
        return {
            "side_frames": [lock_frame, 10],
            "need_action": [True, True],
            "outcome": [0, 0],
            "terminated": [False],
            "truncated": [False],
            "board_bytes": [[0], [0]],
        }, None


def probe() -> TimingProbe:
    return TimingProbe(
        id="p",
        reset_spec={},
        target_side=0,
        column=3,
        row_bottom=5,
        rotation=0,
        lock_frames=(10, 11),
        stratum="mid",
    )


def test_earliest_lock_report_separates_transition_clock_and_value() -> None:
    report = EarliestLockExperiment(Backend(), value_epsilon=0.1).run([probe()])
    assert report.changed_probes == 1
    assert report.clock_divergent_probes == 1
    assert report.beneficial_delays == 1
    assert report.by_stratum["mid"]["changed"] == 1


def test_clock_only_divergence_is_not_structural_change() -> None:
    report = EarliestLockExperiment(ClockOnlyBackend()).run([probe()])
    assert report.changed_probes == 0
    assert report.clock_divergent_probes == 1
