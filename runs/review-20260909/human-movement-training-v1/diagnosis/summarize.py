"""Score and per-placement frame decomposition for ablation journals."""
import json, sys, collections
from pathlib import Path
def summarize(paths):
    out = []
    for path in paths:
        by = collections.defaultdict(list)
        for line in Path(path).read_text().splitlines():
            r = json.loads(line); by[r['comparison']].append(r)
        for c, rows in by.items():
            n = len(rows); score = sum(r['score'] for r in rows) / n
            # side-swapped seed-pair SE
            pairs = collections.defaultdict(list)
            for r in rows: pairs[r['seed']].append(r['score'])
            import statistics as st
            m = [sum(v) / len(v) for v in pairs.values()]
            se = st.pstdev(m) / len(m) ** .5 if len(m) > 1 else 0
            def dec(key):
                t = collections.Counter()
                for r in rows: t.update(r[key])
                d = max(t['decisions'], 1)
                return dict(wait=t['spawn_wait_frames'] / d, steer=t['steering_frames'] / d,
                            descent=(t['execution_frames'] - t['steering_frames']) / d,
                            total=(t['spawn_wait_frames'] + t['execution_frames']) / d,
                            hes=t['human_hesitation_frames'] / d, pills=t['decisions'] / n,
                            fallback=(t['human_route_planner_retimed'] + t['human_route_planner']) / d)
            a, b = dec('a_stats'), dec('b_stats')
            out.append((c, n, score, se, a, b))
            print(f"{c:16s} n={n:3d} score={score:.3f}±{1.96*se:.3f} | A wait {a['wait']:5.1f} steer {a['steer']:5.1f} "
                  f"desc {a['descent']:5.1f} tot {a['total']:5.1f} (hes {a['hes']:4.1f}, fb {a['fallback']:.3f}) | "
                  f"B wait {b['wait']:5.1f} steer {b['steer']:5.1f} desc {b['descent']:5.1f} tot {b['total']:5.1f} | "
                  f"pills/game A {a['pills']:.0f} B {b['pills']:.0f}")
    return out
if __name__ == '__main__': summarize(sys.argv[1:])
