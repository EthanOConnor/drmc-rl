from types import SimpleNamespace

import tools.trainer_event_rollout as events
import tools.trainer_frame_prefetch as prefetch


class _Runner:
    calls = []

    def bfs_full(self, columns, spawn, **kwargs):
        _Runner.calls.append(spawn)
        return SimpleNamespace(copy=lambda: SimpleNamespace(
            costs_u16=events.np.zeros(1, events.np.uint16), offsets_u16=events.np.zeros(1, events.np.uint16),
            lengths_u16=events.np.zeros(1, events.np.uint16), script_buf=events.np.zeros(1, events.np.uint8)))


def test_prefetch_submits_each_spawn_once_and_the_runner_reads_the_shared_answer(monkeypatch):
    monkeypatch.setattr(events, "NativeReachabilityRunner", _Runner)
    requests = []

    def plan(planner, state, delay, pace):
        requests.append((state["side"], delay))
        return planner.bfs_full([state["side"]], SimpleNamespace(
            x=3, y=0, rot=0, speed_counter=0, hor_velocity=0, hold_dir=0, rot_hold=0, frame_parity=0, locked=0))

    monkeypatch.setattr(prefetch, "plan_candidates", plan)
    planner = prefetch.PrefetchingPlanner(2)
    states = [SimpleNamespace(falling=True, terminal=False, spawn_id=1, pill_counter_total=1) for _ in range(4)]
    pool = SimpleNamespace(states=states, semantic=lambda side: {"side": side})
    config = {"variants": {"a": {"delay": 8}, "b": {"delay": 4}}}
    match = {"a": "a", "b": "b", "pace": "top_humans"}
    jobs = [(11, 0, 0), (11, 1, 1)]
    planner.prefetch(config, match, jobs, pool)
    planner.prefetch(config, match, jobs, pool)  # same spawns: nothing new
    planner.close()
    assert sorted(requests) == [(0, 8), (1, 6), (2, 6), (3, 8)]
    before = len(_Runner.calls)
    spawn = SimpleNamespace(x=3, y=0, rot=0, speed_counter=0, hor_velocity=0, hold_dir=0, rot_hold=0,
                            frame_parity=0, locked=0)
    planner.bfs_full([2], spawn)
    assert len(_Runner.calls) == before  # served from the shared exact cache
