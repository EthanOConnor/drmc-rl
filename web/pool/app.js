"use strict";

const DATA = window.DRMC_POOL_DATA || { meta: {}, agents: [], pairwise: [], tournaments: [], pools: [] };
const $ = (sel) => document.querySelector(sel);
const fmt = new Intl.NumberFormat("en-US");
const f0 = (x) => x == null ? "—" : Number(x).toFixed(0);
const f1 = (x) => x == null ? "—" : Number(x).toFixed(1);
const pct = (x) => x == null ? "—" : `${(Number(x) * 100).toFixed(1)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[c]));

let view = "ratings";
let sortState = {};

function boot() {
  $("#mastMeta").innerHTML = `
    <b>${fmt.format(DATA.meta.n_games || 0)}</b> games ·
    <b>${fmt.format(DATA.meta.n_agents || 0)}</b> agents ·
    <b>${fmt.format(DATA.meta.n_tournaments || 0)}</b> tournaments<br>
    built ${esc(DATA.meta.built_at || "unknown")}
  `;
  document.querySelectorAll("#nav button").forEach(btn => {
    btn.addEventListener("click", () => {
      view = btn.dataset.view;
      document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b === btn));
      render();
    });
  });
  render();
}

function kpis(items) {
  return `<div class="kpis">${items.map(([value, label]) => `
    <div class="kpi"><div class="value">${value}</div><div class="label">${label}</div></div>
  `).join("")}</div>`;
}

function table(rows, cols, id, defaultSort, ascDefault = false) {
  const state = sortState[id] || { key: defaultSort, asc: ascDefault };
  const col = cols.find(c => c.key === state.key) || cols[0];
  const val = (r) => col.sort ? col.sort(r) : r[col.key];
  const sorted = [...rows].sort((a, b) => {
    const av = val(a), bv = val(b);
    let cmp = 0;
    if (typeof av === "string" || typeof bv === "string") cmp = String(av ?? "").localeCompare(String(bv ?? ""));
    else cmp = (av ?? -Infinity) - (bv ?? -Infinity);
    return state.asc ? cmp : -cmp;
  });
  const html = `<div class="table-wrap"><table><thead><tr>${cols.map(c => `
    <th class="${c.cls || ""}" data-k="${esc(c.key)}">${esc(c.label)}</th>`).join("")}
    </tr></thead><tbody>${sorted.map(r => `<tr>${cols.map(c => `
      <td class="${c.cls || ""}">${c.fmt ? c.fmt(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
  setTimeout(() => {
    document.querySelectorAll(`#${id} th`).forEach(th => th.onclick = () => {
      const key = th.dataset.k;
      sortState[id] = { key, asc: state.key === key ? !state.asc : false };
      render();
    });
  }, 0);
  return `<div id="${id}">${html}</div>`;
}

function renderRatings() {
  const q = ($("#agentFilter")?.value || "").toLowerCase();
  const comp = $("#componentFilter")?.value || "all";
  const rows = DATA.agents.filter(a => {
    const okQ = !q || a.name.toLowerCase().includes(q) || String(a.checkpoint || "").toLowerCase().includes(q);
    const okC = comp === "all" || String(a.component) === comp;
    return okQ && okC;
  });
  const comps = [...new Set(DATA.agents.map(a => a.component))].sort((a, b) => a - b);
  const latest = DATA.meta.latest_tournament || "none";
  $("#main").innerHTML = `
    <section class="panel">
      <h2>Static Agent Ratings</h2>
      ${kpis([
        [fmt.format(DATA.meta.n_games || 0), "rated games"],
        [fmt.format(DATA.meta.n_agents || 0), "agents"],
        [fmt.format(DATA.meta.n_components || 0), "rating components"],
        [esc(latest), "latest tournament"],
      ])}
      <div class="note">Ratings are static fits over frozen agents. Component numbers mark disconnected result graphs; ratings from different components are not directly comparable.</div>
      <div class="controls">
        <label>filter <input id="agentFilter" type="text" value="${esc(q)}"></label>
        <label>component <select id="componentFilter">
          <option value="all">all</option>
          ${comps.map(c => `<option value="${c}" ${String(c) === comp ? "selected" : ""}>${c}</option>`).join("")}
        </select></label>
      </div>
      ${table(rows, ratingCols(), "ratingTable", "rating")}
    </section>
    <section class="panel">
      <h2>Rating Spread</h2>
      <div id="ratingChart" class="chart"></div>
    </section>
  `;
  $("#agentFilter").addEventListener("input", renderRatings);
  $("#componentFilter").addEventListener("change", renderRatings);
  ratingChart($("#ratingChart"), rows.slice().sort((a, b) => b.rating - a.rating).slice(0, 18));
}

function ratingCols() {
  return [
    { key: "name", label: "agent", cls: "l", fmt: r => `<span class="mono">${esc(r.name)}</span>` },
    { key: "rating", label: "rating", fmt: r => `${r.rating >= 0 ? "+" : ""}${f1(r.rating)}` },
    { key: "ci95", label: "±95", fmt: r => r.ci95 == null ? "—" : f1(r.ci95) },
    { key: "games", label: "games", fmt: r => fmt.format(r.games) },
    { key: "score_rate", label: "score", fmt: r => pct(r.score_rate) },
    { key: "wins", label: "W-D-L", fmt: r => `${r.wins}-${r.draws}-${r.losses}` },
    { key: "component", label: "comp", fmt: r => `<span class="badge">${r.component}</span>` },
    { key: "mode", label: "mode", fmt: r => esc(r.mode || "plain") },
    { key: "last_seen", label: "last seen", cls: "l", fmt: r => esc(r.last_seen || "") },
  ];
}

function renderMatrix() {
  const q = ($("#pairFilter")?.value || "").toLowerCase();
  const rows = DATA.pairwise.filter(p => !q || p.a.toLowerCase().includes(q) || p.b.toLowerCase().includes(q));
  $("#main").innerHTML = `
    <section class="panel">
      <h2>Pairwise Records</h2>
      <div class="note">Records are shown from the alphabetically first agent's perspective.</div>
      <div class="controls"><label>filter <input id="pairFilter" type="text" value="${esc(q)}"></label></div>
      ${table(rows, [
        { key: "a", label: "agent A", cls: "l", fmt: r => `<span class="mono">${esc(r.a)}</span>` },
        { key: "b", label: "agent B", cls: "l", fmt: r => `<span class="mono">${esc(r.b)}</span>` },
        { key: "games", label: "games", fmt: r => fmt.format(r.games) },
        { key: "wins_a", label: "A-B-D", fmt: r => `${r.wins_a}-${r.wins_b}-${r.draws}` },
        { key: "win_rate_a", label: "A win%", fmt: r => pct(r.win_rate_a) },
        { key: "ci", label: "Wilson 95", sort: r => r.ci95[1] - r.ci95[0], fmt: r => `${pct(r.ci95[0])}–${pct(r.ci95[1])}` },
        { key: "avg_sec", label: "avg sec", fmt: r => f0(r.avg_sec) },
        { key: "avg_decisions", label: "avg dec", fmt: r => f0(r.avg_decisions) },
      ], "pairTable", "games")}
    </section>
  `;
  $("#pairFilter").addEventListener("input", renderMatrix);
}

function renderTournaments() {
  $("#main").innerHTML = `
    <section class="panel">
      <h2>Tournaments</h2>
      ${table(DATA.tournaments, [
        { key: "id", label: "#", fmt: r => r.id },
        { key: "name", label: "name", cls: "l", fmt: r => `<span class="mono">${esc(r.name)}</span>` },
        { key: "created", label: "created", cls: "l", fmt: r => esc(r.created.replace("T", " ").replace("+00:00", "Z")) },
        { key: "entries", label: "agents", fmt: r => r.entries.length },
        { key: "games_recorded", label: "games", fmt: r => `${fmt.format(r.games_recorded)} / ${fmt.format(r.games_expected)}` },
        { key: "complete", label: "status", fmt: r => r.complete ? `<span class="pos">complete</span>` : `<span class="neg">running</span>` },
        { key: "top", label: "leader", cls: "l", fmt: r => r.top?.length ? `<span class="mono">${esc(r.top[0].name)}</span> ${r.top[0].rating >= 0 ? "+" : ""}${f1(r.top[0].rating)}` : "—" },
      ], "tournamentTable", "id")}
    </section>
  `;
}

function renderPools() {
  $("#main").innerHTML = `
    <section class="panel">
      <h2>Opponent Pools</h2>
      <div class="note">Pool records are learner-perspective: high score rate means the current learner has been beating that frozen opponent.</div>
      ${table(DATA.pools, [
        { key: "run", label: "run", cls: "l", fmt: r => `<span class="mono">${esc(r.run)}</span>` },
        { key: "updated", label: "updated", cls: "l", fmt: r => esc(r.updated.replace("T", " ").replace("+00:00", "Z")) },
        { key: "entries", label: "pool", fmt: r => r.entries.length },
        { key: "step", label: "step", sort: r => r.latest_skill?.step || 0, fmt: r => fmt.format(r.latest_skill?.step || 0) },
        { key: "whr", label: "grade", sort: r => r.latest_skill?.whr || -Infinity, fmt: r => f0(r.latest_skill?.whr) },
        { key: "cur", label: "clear%", sort: r => r.latest_skill?.cur || -Infinity, fmt: r => pct(r.latest_skill?.cur) },
        { key: "garbage", label: "garbage/min", sort: r => r.latest_skill?.garbage_per_min || -Infinity, fmt: r => f1(r.latest_skill?.garbage_per_min) },
      ], "poolRunTable", "updated")}
    </section>
    <section class="panel">
      <h2>Pool Entries</h2>
      ${table(poolEntries(), [
        { key: "run", label: "run", cls: "l", fmt: r => `<span class="mono">${esc(r.run)}</span>` },
        { key: "id", label: "entry", cls: "l", fmt: r => `<span class="mono">${esc(r.id)}</span>` },
        { key: "games", label: "games", fmt: r => fmt.format(r.games) },
        { key: "learner_score_rate", label: "learner score", fmt: r => `${pct(r.learner_score_rate)} <span class="bar"><span style="width:${Math.max(0, Math.min(100, (r.learner_score_rate || 0) * 100))}%"></span></span>` },
        { key: "wins", label: "learner W", fmt: r => f1(r.wins) },
        { key: "flags", label: "flags", cls: "l", fmt: r => `${r.protected ? "<span class='badge'>protected</span> " : ""}${r.league_target ? "<span class='badge'>target</span>" : ""}` },
      ], "poolEntryTable", "games")}
    </section>
  `;
}

function poolEntries() {
  return DATA.pools.flatMap(p => p.entries.map(e => ({ ...e, run: p.run })));
}

function renderMethod() {
  $("#main").innerHTML = `
    <article class="method">
      <h2>Method</h2>
      <p>
        Agents in these tournaments are frozen checkpoints or fixed policies. The page therefore fits one static
        latent skill per agent with a Bradley-Terry logistic model, draws counted as half points. There is no
        time-varying latent skill or inactivity drift.
      </p>
      <p>
        Re-exporting the page can still move an agent's rating because the estimate conditions on the available
        result graph. New opponents, more games, and a different connected component change the posterior estimate;
        that is estimation movement, not modeled skill drift.
      </p>
      <ul>
        <li>Ratings are centered to mean zero inside each connected component of the result graph.</li>
        <li>Uncertainty is the 95% interval from the observed Fisher information pseudo-inverse.</li>
        <li>Ratings from disconnected components are shown with component tags and should not be compared directly.</li>
        <li>Pool-manifest records are separate learner-perspective training records, not neutral tournament games.</li>
      </ul>
      <h3>What differs from the Fightcade page</h3>
      <p>
        The Fightcade explorer uses a whole-history model because human player strength may change over calendar time.
        This page uses a static model because an internal frozen checkpoint does not improve after it is saved.
        If we want training-progress charts, they should be labeled as checkpoint lineage or estimate snapshots,
        not as a drifting latent skill path for a single agent.
      </p>
    </article>
  `;
}

function ratingChart(el, rows) {
  if (!rows.length) { el.innerHTML = "<div class='note'>No agents in this filter.</div>"; return; }
  const width = Math.max(720, el.clientWidth || 720);
  const rowH = 24;
  const margin = { l: 190, r: 35, t: 15, b: 24 };
  const height = margin.t + margin.b + rowH * rows.length;
  const vals = rows.flatMap(r => [r.rating - (r.ci95 || 0), r.rating + (r.ci95 || 0)]);
  const min = Math.min(...vals, -50), max = Math.max(...vals, 50);
  const x = (v) => margin.l + ((v - min) / (max - min || 1)) * (width - margin.l - margin.r);
  const y = (i) => margin.t + i * rowH + rowH / 2;
  el.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <line class="gridline" x1="${x(0)}" x2="${x(0)}" y1="${margin.t}" y2="${height - margin.b}"></line>
    ${rows.map((r, i) => `
      <text x="${margin.l - 8}" y="${y(i) + 4}" text-anchor="end">${esc(r.name)}</text>
      ${r.ci95 == null ? "" : `<line x1="${x(r.rating - r.ci95)}" x2="${x(r.rating + r.ci95)}" y1="${y(i)}" y2="${y(i)}" stroke="#8d8372" stroke-width="2"></line>`}
      <circle cx="${x(r.rating)}" cy="${y(i)}" r="4" fill="#8a2f2b"></circle>
    `).join("")}
    <line class="axis" x1="${margin.l}" x2="${width - margin.r}" y1="${height - margin.b}" y2="${height - margin.b}"></line>
    <text x="${margin.l}" y="${height - 5}">${f0(min)}</text>
    <text x="${x(0)}" y="${height - 5}" text-anchor="middle">0</text>
    <text x="${width - margin.r}" y="${height - 5}" text-anchor="end">${f0(max)}</text>
  </svg>`;
}

function render() {
  if (view === "ratings") renderRatings();
  else if (view === "matrix") renderMatrix();
  else if (view === "tournaments") renderTournaments();
  else if (view === "pools") renderPools();
  else renderMethod();
}

boot();
