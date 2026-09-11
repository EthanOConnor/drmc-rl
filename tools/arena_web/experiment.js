const experimentPanel = document.querySelector('#experiment');
const resultsPanel = document.querySelector('#experiment-results');
let lastExperiment = '';
const safeNumber = value => Number.isFinite(Number(value)) ? Number(value) : 0;
const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';

function ratingViews(groups, name, openMatrices) {
  return groups.map(group => {
    const key = JSON.stringify([group.label, group.level, group.pace, group.execution_key]);
    const ids = group.ratings.map(r => r.id);
    const signed = n => n > 0 ? `+${n}` : `${n}`;
    return `<div class="section-title"><h3>Relative Elo · ${esc(group.label)}</h3><p>L${group.level} HI · ${esc(group.pace)}</p></div>
      <p class="experiment-note">${esc(name(group.anchor))} = 0. These compare the players in this experiment; they are not drmariostats ratings. Intervals are approximate and count each side-swapped seed as one effective observation.</p>
      <div class="table-wrap"><table><thead><tr><th>Player</th><th>Relative Elo</th><th>95% interval</th><th>Games</th></tr></thead><tbody>
      ${group.ratings.map(r => `<tr><td><strong>${esc(name(r.id))}</strong></td><td>${signed(r.elo)}</td><td>${signed(r.low)} to ${signed(r.high)}</td><td>${r.games}</td></tr>`).join('')}</tbody></table></div>
      <details class="matchup-matrix" data-group="${esc(key)}" ${openMatrices.has(key) ? 'open' : ''}><summary>Matchup scores and coverage</summary><div class="table-wrap"><table><thead><tr><th>Row player vs column</th>${ids.map(id => `<th>${esc(name(id))}</th>`).join('')}</tr></thead><tbody>
      ${ids.map(a => `<tr><th>${esc(name(a))}</th>${ids.map(b => {
        if (a === b) return '<td>—</td>';
        const m = group.matchups.find(m => m.a === a && m.b === b || m.a === b && m.b === a);
        return m ? `<td>${percent(m.a === a ? m.score : 1-m.score)}<span class="sub">${m.games} games</span></td>` : '<td>Unplayed</td>';
      }).join('')}</tr>`).join('')}</tbody></table></div></details>`;
  }).join('');
}

function experimentView(data) {
  experimentPanel.hidden = !data.active;
  resultsPanel.hidden = !data.active;
  for (const selector of ['.hero', '#training-run', '#ratings-status', '#scheduler-status', '#events']) {
    document.querySelector(selector).closest('section').hidden = !!data.active;
  }
  if (!data.active) return;
  const results = data.results || {};
  const training = data.training || {};
  const health = data.health || {};
  const alert = ['failed', 'stale'].includes(health.severity);
  const age = health.training_age_seconds;
  const freshness = age == null ? 'No training timestamp' : age < 60 ? 'Training updated less than a minute ago' : `Training last updated ${Math.floor(age / 60)} min ago`;
  const renderKey = JSON.stringify([data.title, data.status, data.current_work, data.updated_at,
    data.hypothesis, data.goals, data.stages, data.variants, data.findings, data.training, data.pipeline, health.status, health.message, freshness, results]);
  if (renderKey === lastExperiment) return;
  lastExperiment = renderKey;
  experimentPanel.innerHTML = `
    <div class="experiment-heading"><p class="eyebrow">ACTIVE EXPERIMENT</p><span class="badge ${alert ? 'experiment-status-alert' : ''}">${esc(health.status || data.status || 'Preparing')}</span></div>
    <h2>${esc(data.title)}</h2><p>${esc(data.hypothesis)}</p>
    ${alert ? `<div class="experiment-warning" role="alert"><strong>${esc(health.status)}</strong><p>${esc(health.message)}</p><small>${esc(freshness)} · ${safeNumber(training.frames).toLocaleString()} reported frames</small></div>` : ''}
    <div class="current-work"><span>${alert ? 'PLANNED WORK' : 'WORKING ON'}</span><strong>${esc(data.current_work)}</strong></div>
    <div class="experiment-stages">${(data.stages || []).map((stage, i) => `
      <div class="experiment-stage ${['active', 'complete', 'pending'].includes(stage.status) ? stage.status : 'pending'}">
        <span>${stage.status === 'complete' ? '✓' : i + 1}</span><div><strong>${esc(stage.label)}</strong><small>${esc(stage.detail || stage.status)}</small></div>
      </div>`).join('')}</div>
    <div class="brief-grid"><div><strong>Tournament goals</strong><ul>${(data.goals || []).map(goal => `<li>${esc(goal)}</li>`).join('')}</ul></div>
    <div><strong>Findings so far</strong><ul>${(data.findings?.length ? data.findings : ['No candidate results yet. Measurements and findings will appear here.']).map(finding => `<li>${esc(finding)}</li>`).join('')}</ul></div></div>
    <p class="experiment-age">Work update: ${data.updated_at ? esc(new Date(data.updated_at).toLocaleString()) : '—'} · results update: ${results.updated_at ? esc(new Date(results.updated_at).toLocaleString()) : 'No tournament started'}</p>`;
  const variants = data.variants || [];
  const name = id => variants.find(v => v.id === id)?.name || id;
  const tournaments = results.tournaments || [];
  const total = tournaments.reduce((sum, t) => sum + safeNumber(t.played), 0);
  const target = tournaments.reduce((sum, t) => sum + safeNumber(t.target), 0);
  const openMatrices = new Set([...resultsPanel.querySelectorAll('details[open]')].map(el => el.dataset.group));
  resultsPanel.innerHTML = `
    ${data.pipeline?.status ? `<p class="experiment-note"><strong>Study: ${esc(data.pipeline.status)}</strong> · ${safeNumber(data.pipeline.evaluations_complete)} / ${safeNumber(data.pipeline.evaluations_total)} continuation tournament workers complete${data.pipeline.error ? ` · ${esc(data.pipeline.error)}` : ''}</p>` : ''}
    ${training.status ? `<div class="section-title"><h3>Pace-conditioned training</h3><p>${esc(training.status)}</p></div>
      <div class="metrics"><div><span>Simulated console frames</span><strong>${safeNumber(training.frames).toLocaleString()}</strong><small>${training.target_frames ? `Goal: ${safeNumber(training.target_frames).toLocaleString()} frames` : `${training.updates} / ${training.target_updates} updates`}</small></div>
      <div><span>Learning decisions</span><strong>${safeNumber(training.decisions).toLocaleString()}</strong><small>${safeNumber(training.games).toLocaleString()} completed games${training.minimum_decisions_per_pace ? ` · minimum ${safeNumber(training.minimum_decisions_per_pace).toLocaleString()} decisions at every pace` : ''}</small></div>
      <div><span>${training.status === 'Running' && !alert ? 'Collecting' : 'Last collection'} · ${esc(training.current_pace || 'Starting')}</span><strong>${safeNumber(training.collecting_games)} / ${safeNumber(training.collecting_target)} games</strong><small>L${training.current_level || 14} HI · ${safeNumber(training.updates)} updates complete${training.error ? ` · ${esc(training.error)}` : ''}</small></div>
      ${training.throughput ? `<div><span>Latest update throughput</span><strong>${Math.round(safeNumber(training.throughput.frames_per_second)).toLocaleString()} frames/s</strong><small>${safeNumber(training.throughput.learning_decisions_per_second).toFixed(1)} learner decisions/s · includes optimizer</small></div>` : ''}</div>
      ${training.throughput?.breakdown ? `<p class="experiment-note">Latest update time: ${Object.entries(training.throughput.breakdown).map(([phase, seconds]) => `${esc(phase.replaceAll('_seconds', '').replaceAll('_', ' '))} ${safeNumber(seconds).toFixed(1)}s`).join(' · ')}.</p>` : ''}
      <p class="experiment-note">${esc(freshness)}${training.updated_at ? ` · ${esc(new Date(training.updated_at).toLocaleString())}` : ''}. Frame totals advance after each completed optimizer update.</p>
      ${Object.keys(training.paces || {}).length ? `<div class="table-wrap"><table><thead><tr><th>Pace</th><th>Learning decisions</th><th>Console frames</th><th>Completed games</th><th>Games per update</th></tr></thead><tbody>${Object.entries(training.paces).map(([pace, counts]) => `<tr><td>${esc(pace.replaceAll('_', ' '))}</td><td>${counts.learning_decisions == null ? 'Not recorded' : safeNumber(counts.learning_decisions).toLocaleString()}${training.minimum_decisions_per_pace ? `<progress value="${safeNumber(counts.learning_decisions)}" max="${safeNumber(training.minimum_decisions_per_pace)}"></progress>` : ''}</td><td>${counts.frames == null ? 'Not recorded' : safeNumber(counts.frames).toLocaleString()}</td><td>${safeNumber(counts.games).toLocaleString()}</td><td>${training.games_per_pace?.[pace] || '—'}</td></tr>`).join('')}</tbody></table></div>` : ''}
      <p class="experiment-note">Frames measure simulated time, as in the earlier trainer's step counter. Learning decisions count actual selected placements from natural-terminal games; uncontrolled falls and unfinished games supply no learning targets. Both the frame budget and every pace's decision minimum must be reached. Only separate held-out tournaments below establish strength.</p>` : ''}
    <div class="section-title"><h3>${esc(data.variant_title || 'Planning candidates')}</h3><p>${esc(data.comparison_note || 'Same competitive model · execution changes')}</p></div>
    <div class="variant-grid">${variants.map(v => `<article class="variant-card"><span class="badge">${esc(v.status || 'Proposed')}</span><h4>${esc(v.name)}</h4><p>${esc(v.description)}</p></article>`).join('')}</div>
    <div class="section-title"><h3>Tournament progress</h3><p>${total.toLocaleString()}${target ? ` / ${target.toLocaleString()}` : ''} games</p></div>
    <div class="table-wrap"><table><thead><tr><th>Comparison</th><th>Conditions</th><th>Progress</th><th>W / L / D</th><th>Score · 95% interval</th></tr></thead>
    <tbody>${tournaments.map(t => {
      const played = safeNumber(t.played), goal = safeNumber(t.target);
      return `<tr><td><strong>${esc(name(t.a))}</strong><span class="sub">vs ${esc(name(t.b))} · ${esc(t.phase || 'screen')}</span></td>
      <td>L${esc(t.level)} · ${esc(t.speed || 'HI')}<span class="sub">${esc(t.pace || 'Frame Perfect')}</span></td>
      <td>${played.toLocaleString()} / ${goal.toLocaleString()}<progress value="${played}" max="${Math.max(1, goal)}"></progress><span class="sub">${esc(t.status || (played >= goal && goal ? 'Complete' : played ? 'Running' : 'Pending'))}</span></td>
      <td>${safeNumber(t.wins)} / ${safeNumber(t.losses)} / ${safeNumber(t.draws)}</td>
      <td>${played ? percent((safeNumber(t.wins) + .5 * safeNumber(t.draws)) / played) : '—'}<span class="sub">${t.score_ci ? t.score_ci.map(percent).join('–') : 'Awaiting complete seed pairs'}</span></td></tr>`;
    }).join('') || '<tr><td colspan="5">Held-out match results will appear here when the first candidates are ready.</td></tr>'}</tbody></table></div>
    <p class="experiment-note">Score includes draws as half a point. Intervals group the two sides of each seed; live results are provisional. Select a recorded game below to inspect its play.</p>
    ${ratingViews(results.rating_groups || [], name, openMatrices)}
    ${(results.metrics || []).length ? `<div class="section-title"><h3>Execution measurements</h3></div><div class="metrics">${results.metrics.map(m => `<div><span>${esc(m.label)}</span><strong>${esc(m.value)}</strong><small>${esc(m.detail || '')}</small></div>`).join('')}</div>` : ''}`;
}

async function refreshExperiment() {
  try {
    const response = await fetch('/api/experiment', {signal: AbortSignal.timeout(8000)});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    experimentView(await response.json());
    document.querySelector('#experiment-error')?.remove();
  } catch (error) {
    let warning = document.querySelector('#experiment-error');
    if (!warning) {
      warning = document.createElement('p');
      warning.id = 'experiment-error';
      warning.className = 'experiment-warning';
      experimentPanel.before(warning);
    }
    warning.textContent = 'Experiment feed unavailable. Displayed work and results may be stale; reconnecting…';
  } finally {
    setTimeout(refreshExperiment, 5000);
  }
}
refreshExperiment();
