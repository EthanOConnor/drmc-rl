"""Static live dashboard for the showy-knob sweep (``knob.html`` + ``knob.json``).

    nice -n 10 python3 tools/showy_knob/dashboard.py --arms arms.json --out DIR [--loop 60]
    cd DIR && nice -n 10 python3 -m http.server 8110 --bind 127.0.0.1

``arms.json``: ``{"arms": [{"base": "bigclear-champ f100M", "lam": 1.5, "opponent": "self" | "champion",
"parts": [{"file": ".../bc_l15_f.json", "lam": 1.5, "paces": [...], "pairs": 88}, ...]}, ...],
"references": [{"name": ..., "t1_plus": ..., ...}]}``. Each part is a ``tools.showy_knob_h2h``
output (rewritten after every batch), so partial points show as they grow. Pure stdlib; no
pool access; small memory.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import time
from pathlib import Path

W = dict(sloth=0.5, relaxed=0.5, normal=1, fast=1.5, top_humans=2, super_human=3, frame_perfect=3)
FAST4 = ("fast", "top_humans", "super_human", "frame_perfect")
METRICS = ("clears", "t1_plus", "t2_plus", "t3_plus", "horizontal", "horizontal_combo")
ATTACK = ("attacks", "garbage", "quads", "waste")   # present in runs made after the attack counters landed


def elo(q):
    return -400 * math.log10(1 / q - 1) if 0 < q < 1 else float("nan")


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def pooled(per):
    """Pace-weighted outcome and style over ``{pace: h2h result row}``."""
    if not per:
        return None
    wsum = sum(W[p] for p in per)
    score = sum(W[p] * r["knob_score"] for p, r in per.items()) / wsum
    se = math.sqrt(sum((W[p] / wsum) ** 2 * max(r["knob_score"] * (1 - r["knob_score"]), 1e-4) / r["games"]
                       for p, r in per.items()))
    out = dict(games=sum(r["games"] for r in per.values()), elo=elo(score),
               lo=elo(max(score - 1.96 * se, 1e-4)), hi=elo(min(score + 1.96 * se, 1 - 1e-4)))
    for side in ("knob", "base"):
        s = {m: sum(W[p] * r["style"][side][m] for p, r in per.items()) / wsum for m in METRICS}
        s["hshare"] = s["horizontal"] / s["clears"] if s["clears"] else None
        have = all(m in r["style"][side] for r in per.values() for m in ATTACK)
        for m in ATTACK:
            s[m] = sum(W[p] * r["style"][side][m] for p, r in per.items()) / wsum if have else None
        s["quad_share"] = s["quads"] / s["attacks"] if have and s["attacks"] else None
        s["n"] = {m: round(sum(r["style"][side][m] * r["style"][side]["placements"] / 100 for r in per.values()))
                  for m in ("t1_plus", "t2_plus", "t3_plus")}
        bests = [r["style"][side].get("best") for r in per.values() if r["style"][side].get("best")]
        s["best"] = max(bests, key=lambda b: b["score"]) if bests else None
        out[side] = s
    return out


def part_status(part, now):
    d = load(part["file"])
    log = Path(part["file"]).with_suffix(".log")
    if d is None:
        return "queued" if not log.exists() else ("stopped" if now - log.stat().st_mtime > 600 else "starting"), d
    if d.get("finished"):
        return "done", d
    stale = now - Path(part["file"]).stat().st_mtime > 600
    return ("stopped" if stale else "running"), d


def seconds_per_game(all_rows):
    rate = {}
    for p in W:
        rows = [r for r in all_rows if r["pace"] == p and r.get("games")]
        g = sum(r["games"] for r in rows)
        rate[p] = sum(r["wall_seconds"] for r in rows) / g if g else 3.0
    return rate


def build(spec):
    now = time.time()
    arms, all_rows, queue = [], [], []
    for arm in spec["arms"]:
        per, states = {}, []
        for part in arm["parts"]:
            status, d = part_status(part, now)
            rows = [r for r in (d or {}).get("results", []) if r["lam"] == part["lam"]]
            all_rows += rows
            for r in rows:
                per.setdefault(r["pace"], r)
            done_games = sum(r["games"] for r in rows)
            target = 2 * part["pairs"] * len(part["paces"])
            states.append(status)
            queue.append(dict(name=Path(part["file"]).stem, base=arm["base"], lam=part["lam"], opponent=arm["opponent"],
                              status=status, games=done_games, target=target, paces=part["paces"]))
        f4 = {p: r for p, r in per.items() if p in FAST4}
        arms.append(dict(base=arm["base"], lam=arm["lam"], opponent=arm["opponent"],
                         partial=any(s != "done" for s in states), status=states,
                         fast4=pooled(f4), all=pooled(per),
                         sh=pooled({p: per[p] for p in ("super_human",) if p in per}),
                         fp=pooled({p: per[p] for p in ("frame_perfect",) if p in per}),
                         paces=sorted(per, key=list(W).index)))
    # ETA: running parts first, then queued in order, over the configured slots
    rate = seconds_per_game(all_rows)
    slots = [now] * int(spec.get("slots", 2))
    for q in sorted(queue, key=lambda q: {"running": 0, "starting": 0, "queued": 1}.get(q["status"], 2)):
        if q["status"] not in ("running", "starting", "queued"):
            continue
        per_pace = q["target"] / len(q["paces"])
        remaining = sum(max(per_pace - 0, 0) * rate[p] for p in q["paces"]) * max(0.0, 1 - q["games"] / q["target"])
        i = slots.index(min(slots))
        slots[i] += remaining
        q["eta"] = time.strftime("%H:%M", time.localtime(slots[i]))
    # lambda-0 rows: the unbiased side of each base model's self-play arms
    zero = {}
    for arm, cfg in zip(arms, spec["arms"]):
        if cfg["opponent"] == "self" and arm["all"]:
            zero.setdefault(arm["base"], []).append(arm)
    # marginal T2/T3 per 0.25 lambda along each self-play curve
    for base in {a["base"] for a in arms}:
        curve = sorted((a for a in arms if a["base"] == base and a["opponent"] == "self" and a["fast4"]),
                       key=lambda a: a["lam"])
        prev = None
        for a in curve:
            if prev is not None:
                step = (a["lam"] - prev["lam"]) / 0.25
                a["marginal"] = {m: (a["fast4"]["knob"][m] - prev["fast4"]["knob"][m]) / step
                                 for m in ("t2_plus", "t3_plus")}
            prev = a
    return dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), arms=arms, queue=queue,
                references=spec.get("references", []), zero={b: max(v, key=lambda a: (a["fast4"] is not None, a["all"]["games"]))
                         .get("fast4" if any(x["fast4"] for x in v) else "all")["base"] for b, v in zero.items()})


POOL_BASES = ("bigclear-champ-f00100000000", "armA-ppo-v1-f00100000000", "champion-retention-mixed-v2")


def pool_rows(url, condition_set="l14-spawn"):
    """Knob entrants (and their bases) from the rating pool's report.json: per-pace ratings and style."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rep = json.loads(r.read())
    except Exception as error:  # the dashboard never fails on the pool
        return dict(error=f"{type(error).__name__}: {error}")
    cs = next((c for c in rep.get("condition_sets", []) if c["name"] == condition_set), None)
    st = next((x for x in rep.get("style", {}).get("sets", []) if x["condition_set"] == condition_set), None)
    if cs is None:
        return dict(error=f"no condition set {condition_set}")
    wanted = lambda e: "+showy" in e or e in POOL_BASES  # noqa: E731
    per = {}
    for pace in cs["paces"]:
        for r in pace["ratings"]:
            if wanted(r["entrant"]):
                per.setdefault(r["entrant"], {})[pace["pace"]] = r
    style = {e["entrant"]: e for e in (st or {}).get("entrants", []) if wanted(e["entrant"])}
    rows = []
    for entrant in sorted(set(per) | set(style)):
        paces = per.get(entrant, {})
        f4 = {p: r for p, r in paces.items() if p in FAST4}
        def wmean(d):
            if not d:
                return None
            ws = sum(W[p] for p in d)
            m = sum(W[p] * r["rating"] for p, r in d.items()) / ws
            se = math.sqrt(sum((W[p] / ws * r["se"]) ** 2 for p, r in d.items()))
            return dict(elo=m, lo=m - 1.96 * se, hi=m + 1.96 * se, games=sum(r["games"] for r in d.values()))
        rows.append(dict(entrant=entrant, fast4=wmean(f4), all=wmean(paces),
                         sh=wmean({p: paces[p] for p in ("super_human",) if p in paces}),
                         fp=wmean({p: paces[p] for p in ("frame_perfect",) if p in paces}),
                         paces=sorted(paces, key=list(W).index), style=style.get(entrant)))
    return dict(generated=rep.get("generated"), rows=rows)


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def elo_cell(p):
    if not p:
        return "–"
    return f"{p['elo']:+.0f} <span class=ci>[{p['lo']:+.0f}, {p['hi']:+.0f}]</span>"


def style_cells(s):
    if not s:
        return "<td>–</td>" * 10
    n = s["n"]
    best = s["best"]
    b = f"{best['score']:.1f} <span class=ci>({best['cells']}c {best['rounds']}r {best['lines']}l)</span>" if best else "–"
    return (f"<td>{fmt(s['t1_plus'])} <span class=ci>({n['t1_plus']})</span></td>"
            f"<td>{fmt(s['t2_plus'])} <span class=ci>({n['t2_plus']})</span></td>"
            f"<td>{fmt(s['t3_plus'], 4)} <span class=ci>({n['t3_plus']})</span></td>"
            f"<td>{fmt(s['hshare'] * 100 if s['hshare'] is not None else None, 1)}%</td>"
            f"<td>{fmt(s['horizontal_combo'], 2)}</td><td>{fmt(s.get('quads'))}</td><td>{fmt(s.get('garbage'), 1)}</td>"
            f"<td>{fmt(s.get('waste'))}</td><td>{fmt(s['quad_share'] * 100 if s.get('quad_share') is not None else None, 1)}%</td>"
            f"<td>{b}</td>")


HEAD = ("<tr><th>λ</th><th>games</th><th>4-pace Elo</th><th>7-pace Elo</th><th>SH Elo</th><th>FP Elo</th>"
        "<th>T1+ (n)</th><th>T2+ (n)</th><th>T3+ (n)</th><th>h-share</th><th>h-combo</th><th>quads</th><th>garbage</th>"
        "<th>waste</th><th>quad share</th><th>best clear</th>"
        "<th>ΔT2+/0.25λ</th><th>ΔT3+/0.25λ</th></tr>")


def arm_row(a):
    if not a["all"]:
        return f"<tr class=partial><td>{a['lam']}</td><td colspan=17>queued</td></tr>"
    m = a.get("marginal") or {}
    tag = " <span class=badge>partial</span>" if a["partial"] else ""
    k = a["fast4"]["knob"] if a["fast4"] else a["all"]["knob"]
    return (f"<tr class={'partial' if a['partial'] else 'done'}><td>{a['lam']}{tag}</td><td>{a['all']['games']} <span class=ci>({len(a['paces'])}/7 paces)</span></td>"
            f"<td>{elo_cell(a['fast4'])}</td><td>{elo_cell(a['all'])}</td><td>{elo_cell(a['sh'])}</td>"
            f"<td>{elo_cell(a['fp'])}</td>{style_cells(k)}"
            f"<td>{fmt(m.get('t2_plus'))}</td><td>{fmt(m.get('t3_plus'), 4)}</td></tr>")


def chart(arms, base):
    pts = sorted((a for a in arms if a["base"] == base and a["opponent"] == "self" and a["fast4"]), key=lambda a: a["lam"])
    if not pts:
        return ""
    w, h, pad = 300, 150, 30
    lams = [0.0] + [a["lam"] for a in pts]
    x = lambda l: pad + (l - 0) / (max(lams) or 1) * (w - 2 * pad)  # noqa: E731
    t2 = [a["fast4"]["knob"]["t2_plus"] for a in pts]
    t2max = max(t2 + [0.3]) * 1.1
    y2 = lambda v: h - pad - v / t2max * (h - 2 * pad)  # noqa: E731
    lo = min([a["fast4"]["lo"] for a in pts] + [-50]); hi = max([a["fast4"]["hi"] for a in pts] + [50])
    ye = lambda v: h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)  # noqa: E731
    s = [f"<svg viewBox='0 0 {w} {h}' class=chart><text x=4 y=14 class=lbl>T2+ /100 (4-pace)</text>",
         f"<line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} class=axis />"]
    s.append("<polyline class=l2 points='" + " ".join(f"{x(a['lam']):.1f},{y2(v):.1f}" for a, v in zip(pts, t2)) + "' />")
    for a, v in zip(pts, t2):
        s.append(f"<circle cx={x(a['lam']):.1f} cy={y2(v):.1f} r=3 class=p2 /><text x={x(a['lam']) - 8:.1f} y={h - pad + 14} class=lbl>{a['lam']}</text>")
    s.append("</svg>")
    s.append(f"<svg viewBox='0 0 {w} {h}' class=chart><text x=4 y=14 class=lbl>4-pace Elo vs unbiased (95% CI)</text>"
             f"<line x1={pad} y1={ye(0):.1f} x2={w - pad} y2={ye(0):.1f} class=axis />"
             f"<line x1={pad} y1={ye(-25):.1f} x2={w - pad} y2={ye(-25):.1f} class=guide />")
    s.append("<polyline class=le points='" + " ".join(f"{x(a['lam']):.1f},{ye(a['fast4']['elo']):.1f}" for a in pts) + "' />")
    for a in pts:
        f = a["fast4"]
        s.append(f"<line x1={x(a['lam']):.1f} x2={x(a['lam']):.1f} y1={ye(f['lo']):.1f} y2={ye(f['hi']):.1f} class=wh />"
                 f"<circle cx={x(a['lam']):.1f} cy={ye(f['elo']):.1f} r=3 class=pe />"
                 f"<text x={x(a['lam']) - 8:.1f} y={h - 4} class=lbl>{a['lam']}</text>")
    s.append(f"<text x={w - pad + 2} y={ye(-25) + 4:.1f} class=lbl>−25</text></svg>")
    return "".join(s)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#d8d8d2;--acc:#2f6fb3;--acc2:#b3522f;--part:#fff6db}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#e8e8e3;--mut:#9a9a93;--line:#34342f;--acc:#7fb0e6;--acc2:#e69a7f;--part:#3a3420}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1400px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 6px}.mut,.ci{color:var(--mut)}.ci{font-size:12px}
table{border-collapse:collapse;width:100%;overflow-x:auto;display:block}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}tr.partial td{background:var(--part)}.badge{font-size:11px;padding:0 5px;border:1px solid var(--mut);border-radius:8px;color:var(--mut)}
.chart{width:300px;height:150px;margin-right:16px}.axis,.guide{stroke:var(--line)}.guide{stroke-dasharray:3 3}.lbl{fill:var(--mut);font-size:10px}
.l2{fill:none;stroke:var(--acc);stroke-width:2}.p2{fill:var(--acc)}.le{fill:none;stroke:var(--acc2);stroke-width:2}.pe{fill:var(--acc2)}.wh{stroke:var(--acc2)}
"""


def pool_table(pool):
    if not pool:
        return ""
    out = ["<h2>Rating pool (source: pool report.json, l14-spawn, champion = 1500)</h2>"]
    if pool.get("error"):
        return out[0] + f"<div class=mut>pool report unavailable: {html.escape(pool['error'])}</div>"
    out.append(f"<div class=mut>pool report {html.escape(str(pool.get('generated')))}; absolute pool ratings (not vs base); "
               "style pooled over all paces the entrant has played; T1+ is the pool's t1p (≥27).</div><table><tr><th>entrant</th>"
               "<th>games</th><th>4-pace rating</th><th>7-pace rating</th><th>SH</th><th>FP</th><th>T1+</th><th>T2+</th>"
               "<th>T3+</th><th>h-share</th><th>quads</th><th>quad share</th><th>waste</th><th>garbage</th><th>best</th></tr>")
    for r in pool["rows"]:
        st = r["style"] or {}
        def cell(p):
            return "–" if not p else f"{p['elo']:.0f} <span class=ci>[{p['lo']:.0f}, {p['hi']:.0f}]</span>"
        best = st.get("best")
        g = r["all"]["games"] if r["all"] else st.get("games", 0)
        out.append(f"<tr><td>{html.escape(r['entrant'])} <span class=badge>pool</span></td><td>{g} <span class=ci>({len(r['paces'])}/7)</span></td>"
                   f"<td>{cell(r['fast4'])}</td><td>{cell(r['all'])}</td><td>{cell(r['sh'])}</td><td>{cell(r['fp'])}</td>"
                   f"<td>{fmt(st.get('t1p'))}</td><td>{fmt(st.get('t2'))}</td><td>{fmt(st.get('t3'), 4)}</td>"
                   f"<td>{fmt(st['horizontal'] * 100, 1) + '%' if st.get('horizontal') is not None else '–'}</td>"
                   f"<td>{fmt(st.get('quads'))}</td><td>{fmt(st.get('quad_share'))}</td><td>{fmt(st.get('waste'))}</td>"
                   f"<td>{fmt(st.get('garbage'), 1)}</td><td>{best['score'] if best else '–'}</td></tr>")
    out.append("</table>")
    return "".join(out)


def render(data):
    arms = data["arms"]
    out = [f"<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=60>"
           f"<meta name=viewport content='width=device-width,initial-scale=1'><title>Showy knob sweep</title><style>{CSS}</style></head><body>",
           "<h1>Showy knob sweep</h1>",
           f"<div class=mut>Updated {data['updated']} · auto-refresh 60 s · knob vs opponent, L14 Hi, delay 4, both sides of each seed. "
           "Headline Elo is the 4-pace weighted mean (fast 1.5, top_humans 2, super_human 3, frame_perfect 3); style columns are "
           "4-pace weighted rates per 100 placements with pooled counts (n). Highlighted rows are still running.</div>"]
    for base in dict.fromkeys(a["base"] for a in arms):
        rows = [a for a in arms if a["base"] == base and a["opponent"] == "self"]
        if not rows:
            continue
        out.append(f"<h2>{html.escape(base)} + knob vs its unbiased self <span class=badge>local h2h</span></h2><div>{chart(arms, base)}</div><table>{HEAD}")
        z = data["zero"].get(base)
        if z:
            out.append(f"<tr><td>0 (unbiased)</td><td>–</td><td>0</td><td>0</td><td>–</td><td>–</td>{style_cells(z)}<td></td><td></td></tr>")
        out += [arm_row(a) for a in sorted(rows, key=lambda a: a["lam"])]
        out.append("</table>")
    vs = [a for a in arms if a["opponent"] != "self"]
    if vs:
        out.append(f"<h2>Head-to-heads vs the plain champion</h2><table>{HEAD.replace('<th>λ</th>', '<th>entrant</th>')}")
        for a in vs:
            out.append(arm_row(dict(a, lam=f"{a['base']} λ{a['lam']}")))
        out.append("</table>")
    out.append(pool_table(data.get("pool")))
    out.append("<h2>Reference (per 100 placements, 14-Hi)</h2><table><tr><th>who</th><th>T1+</th><th>T2+</th><th>T3+</th>"
               "<th>h-share</th><th>h-combo</th><th>quads</th><th>garbage</th><th>waste</th><th>quad share</th></tr>")
    for r in data["references"]:
        out.append(f"<tr><td>{html.escape(r['name'])}</td><td>{fmt(r['t1_plus'])}</td><td>{fmt(r['t2_plus'])}</td>"
                   f"<td>{fmt(r['t3_plus'], 4)}</td><td>{fmt(r['hshare'] * 100 if r.get('hshare') is not None else None, 1)}%</td><td>{fmt(r['horizontal_combo'], 2)}</td>"
                   f"<td>{fmt(r.get('quads'))}</td><td>{fmt(r.get('garbage'), 1)}</td><td>{fmt(r.get('waste'))}</td>"
                   f"<td>{fmt(r['quad_share'] * 100 if r.get('quad_share') is not None else None, 1)}%</td></tr>")
    out.append("</table><h2>Queue</h2><table><tr><th>run</th><th>model</th><th>λ</th><th>vs</th><th>status</th><th>games</th><th>paces</th><th>ETA</th></tr>")
    order = {"running": 0, "starting": 0, "queued": 1, "stopped": 2, "done": 3}
    for q in sorted(data["queue"], key=lambda q: order.get(q["status"], 4)):
        out.append(f"<tr class={'partial' if q['status'] in ('running', 'starting') else ''}><td>{q['name']}</td><td>{html.escape(q['base'])}</td>"
                   f"<td>{q['lam']}</td><td>{q['opponent']}</td><td>{q['status']}</td><td>{q['games']}/{q['target']}</td>"
                   f"<td>{len(q['paces'])}</td><td>{q.get('eta', '')}</td></tr>")
    out.append("</table></body></html>")
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-report", help="rating pool report.json URL (e.g. http://127.0.0.1:8098/report.json)")
    ap.add_argument("--loop", type=float, default=0, help="rewrite every N seconds (0: once)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    while True:
        spec = json.loads(args.arms.read_text())
        data = build(spec)
        if args.pool_report:
            data["pool"] = pool_rows(args.pool_report)
        for name, text in (("knob.json", json.dumps(data, indent=1, default=float)), ("knob.html", render(data))):
            tmp = args.out / (name + ".tmp")
            tmp.write_text(text)
            tmp.replace(args.out / name)
        if not args.loop:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
