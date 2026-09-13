'use strict';
const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => Number.isFinite(Number(value)) ? Number(value) : 0;
const count = value => num(value).toLocaleString();
const signed = value => `${value > 0 ? '+' : ''}${value}`;
const compact = value => num(value) >= 1e6 ? `${(num(value)/1e6).toFixed(2)}M` : num(value) >= 1e3 ? `${(num(value)/1e3).toFixed(1)}k` : count(value);
const paceNames = {sloth:'Sloth',relaxed:'Relaxed',normal:'Normal',fast:'Fast',top_humans:'Top Humans',super_human:'Super Human',frame_perfect:'Frame Perfect'};
const age = value => value ? Math.max(0,(Date.now()-Date.parse(value))/1000) : Infinity;
const ago = value => !Number.isFinite(value) ? 'No update yet' : value < 60 ? 'Updated just now' : value < 3600 ? `${Math.floor(value/60)}m since update` : `${Math.floor(value/3600)}h since update`;
const bar = (value,target) => `<div class="bar"><i style="width:${Math.max(0,Math.min(100,100*num(value)/Math.max(1,num(target))))}%"></i></div>`;
function trainingPhase(t) {
  if(t.status!=='Running') return t.status || 'Waiting';
  return {collecting:'Collecting games',auditing_collection:'Checking collection',optimizing:'Updating network',checking_update:'Checking update',saving_replay:'Saving public replay',saving_checkpoint:'Saving checkpoint',between_updates:'Between updates'}[t.phase] || 'Running';
}
function trainingActivity(t) {
  const w=t.activity;
  if(t.status!=='Running' || !w) return '';
  if(w.phase==='collecting') return `${count(w.games)} / ${count(w.target)} games finished · ${compact(w.frames)} frames simulated · ${compact(w.decision_requests)} decision requests in this update`;
  if(w.phase==='optimizing') return `Epoch ${count(w.epoch)} / ${count(w.epochs)} · step ${count(w.step)} / ${count(w.steps)}${w.attempt>1?` · retry ${count(w.attempt-1)}`:''}`;
  if(w.phase==='auditing_collection' || w.phase==='checking_update') return `${compact(w.checked)} / ${compact(w.total)} decisions checked`;
  return trainingPhase(t);
}
function saved(key,fallback) { try { return localStorage.getItem(`pp-tournament-${key}`) || fallback; } catch { return fallback; } }
function save(key,value) { try { localStorage.setItem(`pp-tournament-${key}`,value); } catch {} }
let experiment = null, snapshot = null, reference = saved('reference','parent'), condition = saved('condition','14:normal');
let feedError = '';
const entrantName = id => experiment?.variants?.find(v=>v.id===id)?.name || snapshot?.agents?.find(v=>v.id===id)?.name || id;
function readyEntrants() {
  const ready = new Set((experiment.results?.entrants || []).filter(v=>v.ready).map(v=>v.id));
  for (const match of experiment.results?.tournaments || []) if (match.played) {ready.add(match.a);ready.add(match.b);}
  for (const run of experiment.training_runs || []) if(run.variant_id && run.status==='Training complete' && run.final_checkpoint) ready.add(run.variant_id);
  return ready;
}
const conditionKey = item => `${item.level}:${item.pace || 'frame_perfect'}${item.execution_key ? ':'+item.execution_key : ''}`;
const conditionName = item => `${item.level} HI · ${paceNames[item.pace] || item.pace}${item.execution_profile ? ` · ${item.execution_profile.reaction_frames}f reaction` : ' · earlier presets'}`;
const replayCondition = item => {
  const comparison = item.match_key?.replace(/-\d+$/,'');
  const match = experiment?.results?.tournaments?.find(m=>m.id===comparison);
  return match ? conditionName(match) : `${item.level ?? '?'} HI`;
};

function setOptions(select,entries,value) {
  const key = JSON.stringify(entries);
  if (select.dataset.options !== key) {
    select.innerHTML = entries.map(e=>`<option value="${esc(e.value)}" ${e.disabled?'disabled':''}>${esc(e.label)}</option>`).join('');
    select.dataset.options = key;
  }
  select.value = value;
}

function renderStandings() {
  const results = experiment.results || {}, groups = results.unified_rating_groups || results.rating_groups || [];
  const conditions = new Map();
  for (const match of results.tournaments || []) conditions.set(conditionKey(match),{level:match.level,pace:match.pace || 'frame_perfect',execution_key:match.execution_key,execution_profile:match.execution_profile});
  if (!conditions.size) conditions.set('14:normal',{level:14,pace:'normal'});
  if (!conditions.has(condition)) condition = conditions.has('14:normal') ? '14:normal' : conditions.keys().next().value;
  const entries = [...conditions].sort((a,b)=>a[1].level-b[1].level || Object.keys(paceNames).indexOf(a[1].pace)-Object.keys(paceNames).indexOf(b[1].pace));
  setOptions($('#conditions'),entries.map(([value,field])=>({value,label:conditionName(field)})),condition);
  const group = groups.find(g=>conditionKey(g)===condition);
  const ratings = group?.ratings || [], rated = new Map(ratings.map(r=>[r.id,r]));
  const variants = experiment.variants || [];
  if (!rated.has(reference)) reference = rated.has(experiment.rating_anchor) ? experiment.rating_anchor : ratings[0]?.id || experiment.rating_anchor || 'parent';
  setOptions($('#reference'),variants.map(v=>({value:v.id,label:entrantName(v.id),disabled:ratings.length > 0 && !rated.has(v.id)})),reference);
  const ready = readyEntrants();
  const ids = [...ratings.map(r=>r.id),...variants.map(v=>v.id).filter(id=>!rated.has(id))];
  $('#ranking').innerHTML = ids.map((id,i)=>{
    const rating = rated.get(id), delta = rating?.differences?.[reference] || (reference === group?.anchor ? rating : null);
    const note = id === reference ? 'Elo reference' : !rating ? ready.has(id) ? 'Awaiting paired results' : 'Checkpoint pending' : rating.games < 256 ? 'Provisional · limited games' : 'Live estimate';
    const decisive = delta && (delta.low > 0 || delta.high < 0);
    return `<tr class="${id===reference?'reference-row':''}"><td class="rank">${rating?i+1:'—'}</td><td class="player-name">${esc(entrantName(id))}<span class="player-note ${id===reference?'reference-label':''}">${note}</span></td><td class="number elo ${decisive ? delta.elo>0?'positive':'negative' : 'unrated'}">${delta?signed(delta.elo):'—'}</td><td class="number interval">${delta?`${signed(delta.low)} to ${signed(delta.high)}`:'Unrated'}</td><td class="number">${rating?count(rating.games):'—'}</td></tr>`;
  }).join('') || '<tr><td colspan="5" class="empty">The first paired results will appear here.</td></tr>';
  const games = ratings.reduce((n,r)=>n+num(r.games),0)/2;
  $('#rating-note').textContent = `${count(games)} games at these settings · approximate 95% intervals · paired seeds. Levels and recorded motor limits are kept separate. Earlier results retain their original grouping.`;
  const matched = group?.matchups || [];
  $('#matrix').innerHTML = ratings.length ? `<table><thead><tr><th>Row vs column</th>${ratings.map((r,i)=>`<th title="${esc(entrantName(r.id))}">${i+1}</th>`).join('')}</tr></thead><tbody>${ratings.map((a,i)=>`<tr><th>${i+1}. ${esc(entrantName(a.id))}</th>${ratings.map(b=>{
    if(a.id===b.id)return '<td>—</td>';
    const m=matched.find(m=>m.a===a.id&&m.b===b.id||m.b===a.id&&m.a===b.id);
    return m?`<td>${(100*(m.a===a.id?m.score:1-m.score)).toFixed(1)}%<small>${count(m.games)} games</small></td>`:'<td>—</td>';
  }).join('')}</tr>`).join('')}</tbody></table>`:'<p class="empty">No paired results at these settings.</p>';
}

function renderOperations() {
  const t = experiment.training || {}, r = experiment.results || {}, workers = r.workers || [];
  const ready = readyEntrants();
  const total = (r.tournaments || []).reduce((n,m)=>n+num(m.played),0);
  const active = workers.filter(w=>w.status==='Playing');
  const decisionBudget = t.target_decisions && !t.target_frames;
  $('#summary').innerHTML = `<div><span>Checkpoints ready</span><strong>${ready.size}</strong><small>/ ${(experiment.variants||[]).length}</small></div><div><span>Tournament games</span><strong>${count(total)}</strong></div><div><span>Training ${decisionBudget?'placements':'frames'}</span><strong>${compact(decisionBudget?t.decisions:t.frames)}</strong><small>/ ${compact(decisionBudget?t.target_decisions:t.target_frames)}</small></div><div><span>Evaluators playing</span><strong>${active.length}</strong><small>/ ${workers.length}</small></div>`;
  $('#worker-count').textContent = `${workers.length} workers`;
  $('#workers').innerHTML = workers.map(w=>{
    const match=(r.tournaments || []).find(m=>m.id===w.current_match);
    const stale = w.status==='Playing' && age(w.updated_at)>900;
    const status=stale?'Update overdue':w.status;
    return `<div class="worker"><div class="worker-top"><strong>${esc(w.host)} · ${esc(w.device?.toUpperCase() || '')}</strong><span class="tag ${w.status==='Failed'||stale?'bad':w.status==='Playing'?'good':''}">${esc(status)}</span></div><div class="worker-match">${match?`${esc(entrantName(match.a))}<br><span>vs</span> ${esc(entrantName(match.b))}`:esc(w.error || (w.status==='Complete'?'Schedule complete':'Waiting for the next frozen checkpoint'))}</div>${match?bar(match.played,match.target):''}<div class="worker-meta"><span>${match?`${esc(conditionName(match))} · ${count(match.played)} / ${count(match.target)}`:''}</span><span>${ago(age(w.updated_at))}</span></div></div>`;
  }).join('') || '<p class="empty">Evaluation workers are starting.</p>';
  $('#training-status').textContent = t.status==='Running'?'Running':t.status || 'Waiting';
  $('#training-status').className = `tag ${t.status==='Failed'?'bad':t.status==='Running'?'good':''}`;
  $('#training').innerHTML = `<div class="training-body"><div class="training-head"><strong>${compact(t.frames)}</strong><span>${t.target_frames?`/ ${compact(t.target_frames)}`:''} simulated frames</span></div>${t.target_frames?bar(t.frames,t.target_frames):''}<div class="training-detail"><span>${compact(t.throughput?.frames_per_second)} frames/s</span><span>${count(t.updates)} updates${t.optimizer_steps!=null?` · ${count(t.optimizer_steps)} optimizer steps`:''}</span></div>${Object.entries(t.paces || {}).map(([pace,c])=>`<div class="pace-budget"><span>${esc(paceNames[pace] || pace)}</span>${bar(c.learning_decisions,t.minimum_decisions_per_pace)}<strong>${compact(c.learning_decisions)}</strong></div>`).join('')}<p class="budget-note">${compact(t.minimum_decisions_per_pace)} learned placements required at every pace.<br>${ago(age(t.updated_at))}${t.status==='Running'?` · ${esc(paceNames[t.current_pace] || t.current_pace)} ${count(t.activity?.phase==='collecting'?t.activity.games:t.collecting_games)}/${count(t.collecting_target)} games`:''}</p></div>`;
  if (t.target_decisions) {
    $('#training').insertAdjacentHTML('afterbegin', `<div class="training-body"><div class="training-head"><strong>${compact(t.decisions)}</strong><span>/ ${compact(t.target_decisions)} learner decisions</span></div>${bar(t.decisions,t.target_decisions)}<div class="training-detail"><span>${compact(t.throughput?.learning_decisions_per_second)} decisions/s</span><span>${esc(trainingPhase(t))}</span></div><p class="budget-note">Budgets count completed updates. ${t.target_frames?'Frame and decision targets':'Overall and per-pace decision targets'} must be met.</p>${trainingActivity(t)?`<p class="budget-note">${esc(trainingActivity(t))}</p>`:''}</div>`);
  }
  if(experiment.training_runs?.length){
    $('#training').insertAdjacentHTML('afterbegin',`<div class="training-body">${experiment.training_runs.map(run=>{const decisions=run.target_decisions&&!run.target_frames;const value=decisions?run.decisions:run.frames,target=decisions?run.target_decisions:run.target_frames;return `<div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)} · ${compact(value)} / ${compact(target)} ${decisions?'placements':'frames'}</span></div>${bar(value,target)}`;}).join('')}<p class="budget-note">Details below: ${esc(t.label)}. Last update KL: ${t.losses?.update_kl==null?'—':Number(t.losses.update_kl).toFixed(4)}.</p></div>`);
  }
  const pipeline = experiment.pipeline || {};
  if(pipeline.schema==='drmc-conditional-next-pill-opportunity-v1'){
    $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>Future movement labels</strong><span>${esc(pipeline.status)}</span></div>${bar(pipeline.roots,pipeline.target_roots)}<div class="training-detail"><span>${count(pipeline.roots)} / ${count(pipeline.target_roots)} positions</span><span>${count(pipeline.candidates)} candidate placements</span></div><p class="budget-note">${count(pipeline.next_candidates)} next-pill placements checked · ${count(pipeline.splits?.validation)} validation positions.<br>Conditional on no incoming garbage. ${ago(age(pipeline.updated_at))}</p></div>`);
  } else if(pipeline.schema==='drmc-motor-auxiliary-study-v1'){
    $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>Future movement fitting</strong><span>${esc(pipeline.status)}</span></div>${bar(pipeline.epoch,pipeline.target_epochs)}<div class="training-detail"><span>${count(pipeline.epoch)} / ${count(pipeline.target_epochs)} epochs</span><span>${count(pipeline.accepted_examples)} accepted examples</span></div><p class="budget-note">${count(pipeline.train_roots)} training positions · ${count(pipeline.validation_roots)} validation positions · ${count(pipeline.anchor_games)} policy-preservation games.<br>${esc(pipeline.phase)} · ${ago(age(pipeline.updated_at))}</p></div>`);
  } else if(pipeline.schema==='drmc-motor-confirmation-v1'){
    $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>Movement prediction confirmation</strong><span>${esc(pipeline.status)}</span></div>${bar(pipeline.completed_conditions,pipeline.target_conditions)}<div class="training-detail"><span>${count(pipeline.completed_conditions)} / ${count(pipeline.target_conditions)} conditions complete</span><span>${esc(pipeline.condition || '')}</span></div><p class="budget-note">${esc(pipeline.phase)} · ${count(pipeline.games)} source games · ${count(pipeline.roots)} labeled positions.<br>Reserved game seeds. Prediction errors are evaluated separately from playing strength. ${ago(age(pipeline.updated_at))}</p></div>`);
  }
  for(const run of experiment.research_runs || []){
    if(['drmc-controller-retention-study-v1','drmc-controller-retention-mixed-study-v2'].includes(run.schema)){
      const replacement=run.schema==='drmc-controller-retention-mixed-study-v2';
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div><p class="budget-note">${esc((run.phase || '').replaceAll('_',' '))} · ${count(run.completed_jobs?.length)} / ${replacement?1:3} stages complete.<br>${replacement?'Replacement mixed trial; completed control retained.':'Frozen teacher games, then the cycling control and mixed-pace retention run.'} ${compact(run.target_decisions_per_arm)} placements per arm, at least ${compact(run.minimum_decisions_per_pace)} at each speed.</p></div>`);
      continue;
    }
    if(run.schema==='drmc-controller-retention-collection-v1'){
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.games,run.target_games)}<div class="training-detail"><span>${count(run.games)} / ${count(run.target_games)} games</span><span>${count(run.anchor_rows)} reference decisions</span></div><div class="training-detail"><span>${count(run.conditions?.length)} / 7 speeds complete</span><span>${esc(paceNames[run.current_pace] || run.current_pace || '')}</span></div><p class="budget-note">Natural games from frozen opponents. These reference decisions are separate from outcome-training data.<br>${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-target-construction-v1'){
      const target=num(run.training_windows)*num(run.config?.epochs);
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.window_presentations,target)}<div class="training-detail"><span>${count(run.action_presentations)} action presentations</span><span>${count(run.epochs?.length)} / ${count(run.config?.epochs)} epochs</span></div><p class="budget-note">Routes are trained for one fixed requested target over up to six placements. The root proposer and competitive core stay frozen. Actual controlled-game results appear separately.<br>${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-target-construction-evaluation-v1'){
      const arms=['controlled','shadow'].map(name=>{
        const arm=run.arms?.[name] || {}, label=name==='controlled'?'Proposal controls':'Competitive controls';
        return `<div class="training-detail"><strong>${label}</strong><span>${count(arm.games)} games</span></div><div class="training-detail"><span>Original targets reached</span><span>${count(arm.verified_original_payoffs)} / ${count(arm.plans)}</span></div><div class="training-detail"><span>After multiple placements</span><span>${count(arm.multi_placement_payoffs)}</span></div>`;
      }).join('');
      const match=run.config?.arena?.schedule?.find(m=>m.id===run.current_condition);
      const comparisons=(run.comparisons || []).map(item=>{
        const condition=run.config?.arena?.schedule?.find(m=>m.id===item.condition), score=item.contrasts?.match_score;
        if(!score) return '';
        return `<div class="training-detail"><span>${esc(condition?conditionName(condition):item.condition)}</span><span>${(100*score.controlled_minus_shadow).toFixed(1)} pp</span></div>`;
      }).join('');
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.games,run.target_games)}<div class="training-detail"><span>${count(run.games)} / ${count(run.target_games)} games</span><span>${esc(match?conditionName(match):'')}</span></div>${arms}${comparisons?`<p class="budget-note">Match score change versus unchanged controls (percentage points)</p>${comparisons}`:''}<p class="budget-note">Experimental proposal actions run in real controller games. Strength cost is measured against unchanged controls on the same seeds; quality admission and blind preferences remain open.<br>${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(['drmc-spatial-execution-audit-v1','drmc-spatial-execution-assessment-v1'].includes(run.schema)){
      const arms=['persistent','stateless'].map(name=>{
        const arm=run.aggregate?.[name] || run.arms?.[name] || {};
        const label=name==='persistent'?'Carried history':'Reset history';
        const delayed=arm.delayed_hits===undefined?'':`<div class="training-detail"><span>First reached after multiple placements</span><span>${count(arm.delayed_hits)}</span></div>`;
        return `<div class="training-detail"><strong>${label}</strong><span>${count(arm.original_target_hits ?? arm.root_goals_observed)} / ${count(arm.plans)} original targets observed</span></div>${delayed}<div class="training-detail"><span>First choice reachable</span><span>${count(arm.raw_preference_reachable ?? arm.raw_preferences_reachable)} / ${count(arm.decisions)} decisions</span></div>`;
      }).join('');
      const match=run.config?.arena?.schedule?.find(m=>m.id===run.current_condition);
      const scope=run.status==='Complete'?'7 paces at 14 HI · 2 at 20 HI':match?conditionName(match):'';
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.games,run.target_games)}<div class="training-detail"><span>${count(run.games)} / ${count(run.target_games)} games</span><span>${esc(scope)}</span></div>${arms}<p class="budget-note">Measured during unchanged competitive play. Later payoff alone does not prove a construction caused it. Strength with proposal control and blind preferences remain unevaluated.<br>${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-neural-preparation-assessment-v1'){
      const t=run.median_ms || {}, ms=k=>`${num(t[k]).toFixed(1)} ms`;
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div><div class="training-detail"><span>Matching conditional choices</span><span>${count(run.previews)} / ${count(run.previews)}</span></div><div class="training-detail"><span>9 previews, fresh inference</span><span>${ms('full_nine_ms')}</span></div><div class="training-detail"><span>Prepare both bottles</span><span>${ms('preparation_ms')}</span></div><div class="training-detail"><span>9 previews, after preparation</span><span>${ms('prepared_nine_ms')}</span></div><div class="training-detail"><span>Preparation + 9 previews</span><span>${ms('prepared_total_nine_ms')}</span></div><div class="training-detail"><span>One decision, fresh / own bottle cached</span><span>${ms('full_one_ms')} / ${ms('own_prepared_fresh_opponent_one_ms')}</span></div><p class="budget-note">Disabling early conditioning changed ${count(run.migration_changed_choices)} of ${count(run.roots)} original choices. This architecture needs student training and strength evaluation before adoption. Shared-Mac timings.</p></div>`);
      continue;
    }
    if(run.schema==='drmc-adaptive-search-audit-v1'){
      const records=run.records || [], record=records.at(-1);
      const unilateral=['p1','p2'].includes(record?.root_boundary);
      const checks=records.flatMap(r=>Object.values(r.comparisons || {}));
      const pending=record?.order?.find(name=>!record.variants?.[name]);
      const completed=records.reduce((n,r)=>n+Object.keys(r.variants || {}).length,0);
      const variants=1+(run.config?.compare_unextended?1:0)+(run.config?.allocation_modes?.length || 1)*(run.config?.allocation_batches?.length || 2);
      const rows=(record?.order || []).map(name=>{
        const result=record.variants?.[name];
        const label=name==='unextended'?'Without extensions':name==='complete'?'Complete reference':name.startsWith('nested-')?'Nested allocation':'Root allocation';
        const status=result?(['complete','unextended'].includes(name)?(result.budget_exhausted?'Budget exhausted':result.equilibrium_converged===false?'Uncertified':'Computed'):(result.certified?(unilateral?'Regret bound met':'Response bound met'):'Uncertified')):(run.status==='Running'&&name===pending?'Evaluating':'Queued');
        return `<div class="training-detail"><strong>${label}</strong><span>${status}</span></div>${result?`<div class="training-detail"><span>${num(result.seconds).toFixed(1)}s</span><span>${count(result.nodes)} native nodes</span></div>${result.evaluated_root_actions!=null?`<div class="training-detail"><span>Root actions evaluated</span><span>${count(result.evaluated_root_actions)} / ${count(result.total_root_actions)}</span></div>`:''}${result.tactical_extensions?`<div class="training-detail"><span>Extended decisions</span><span>${count(result.tactical_extensions)}</span></div>`:''}`:''}`;
      }).join('');
      const extension=run.search_config?.tactical_extension_events?` + up to ${count(run.search_config.tactical_extension_events)} tactical`:'';
      const depth=num(run.search_config?.depth_events);
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(completed,num(run.config?.states)*variants)}<div class="training-detail"><span>${count(completed)} / ${count(num(run.config?.states)*variants)} variant runs</span><span>${count(depth)} event ${depth===1?'level':'levels'}${extension}</span></div>${unilateral?`<div class="training-detail"><span>Position ${count(records.length)} / ${count(run.config?.states)}</span><span>P${num(record.root_side)+1} deciding</span></div>`:''}${rows}${checks.length?`<div class="training-detail"><span>Full-reference bound checks</span><span>${count(checks.filter(c=>c.bounds_hold).length)} / ${count(checks.length)} passed</span></div>`:''}<p class="budget-note">Updates after each variant. Bounds describe the finite-depth critic game; playing strength is untested.</p></div>`);
      continue;
    }
    if(run.schema==='drmc-motor-confirmation-v1'){
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.completed_conditions,run.target_conditions)}<div class="training-detail"><span>${count(run.completed_conditions)} / ${count(run.target_conditions)} conditions complete</span><span>${esc(run.condition || '')}</span></div><p class="budget-note">${esc(run.phase)} · ${count(run.games)} source games · ${count(run.roots)} labeled positions in this condition.<br>Fresh reserved seeds. Prediction accuracy and playing strength are evaluated separately. ${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-motor-auxiliary-study-v1'){
      const heads=num(run.target_head_epochs)>0;
      const presentations=num(run.head_accepted_examples)+num(run.accepted_examples);
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${heads?`<div class="training-detail"><span>Prediction heads</span><span>${count(run.head_epoch)} / ${count(run.target_head_epochs)} epochs</span></div>${bar(run.head_epoch,run.target_head_epochs)}`:''}<div class="training-detail"><span>Shared core</span><span>${count(run.epoch)} / ${count(run.target_epochs)} epochs</span></div>${bar(run.epoch,run.target_epochs)}<div class="training-detail"><span>${count(presentations)} root presentations</span><span>${count(run.anchor_games)} policy-anchor games</span></div><p class="budget-note">${esc((run.phase || '').replaceAll('_',' '))} · ${ago(age(run.updated_at))}<br>${count(run.train_roots)} training roots · ${count(run.validation_roots)} validation roots. No outcome-training frames.</p></div>`);
      continue;
    }
    if(['drmc-spatial-expressive-proposer-v1','drmc-spatial-expressive-confirmation-v1'].includes(run.schema)){
      const confirmation=run.schema==='drmc-spatial-expressive-confirmation-v1';
      const evaluated=run.evaluated_sessions ?? run.paired_session_comparisons?.action_nll?.sessions;
      const target=num(run.training_windows)*num(run.config?.epochs);
      const arms=['persistent','stateless'].map(name=>{
        const arm=run.arms?.[name] || {};
        const label=name==='persistent'?'Persistent plan':'Stateless control';
        if(confirmation)return `<div class="training-detail"><strong>${label}</strong><span>${esc(arm.status || (run.current_arm===name?'Evaluating':'Queued'))}</span></div><div class="training-detail"><span>${count(arm.evaluated_windows)} constructions</span><span>${count(arm.evaluated_actions)} actions evaluated</span></div>`;
        return `<div class="training-detail"><strong>${label}</strong><span>${esc(arm.status || 'Queued')}</span></div>${target?bar(arm.window_presentations,target):''}<div class="training-detail"><span>${count(arm.action_presentations)} action presentations</span><span>${count(arm.epochs?.length)} / ${count(run.config?.epochs)} epochs</span></div>`;
      }).join('');
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${run.phase==='shared_features'?`${bar(run.feature_rows,run.unique_public_inputs)}<p class="budget-note">${count(run.feature_rows)} / ${count(run.unique_public_inputs)} frozen feature inputs</p>`:''}${arms}<p class="budget-note">${esc((run.phase || '').replaceAll('_',' '))} · ${ago(age(run.updated_at))}<br>${confirmation?`${count(run.confirmation_sessions)} reserved sessions${evaluated!=null?` · ${count(evaluated)} with scored constructions`:''} · No training updates.`:'Separate trained controls on recorded human play.'} Strength and preference tests remain.</p></div>`);
      continue;
    }
    if(run.schema==='drmc-expressive-sequences-v1'){
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.sessions,run.target_sessions)}<div class="training-detail"><span>${count(run.sessions)} / ${count(run.target_sessions)} replay sessions</span><span>${count(run.windows)} constructions</span></div><p class="budget-note">${count(run.counters?.verified)} reproduced placements. Sequences stop at garbage, gaps or mismatched boards.<br>Observed geometry; preference and strength are evaluated separately. ${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-persistent-expressive-proposer-v1'){
      const target=num(run.training_windows)*num(run.config?.epochs);
      const latest=run.epochs?.at(-1)?.validation;
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.window_presentations,target)}<div class="training-detail"><span>${count(run.window_presentations)} / ${count(target)} construction presentations</span><span>${count(run.epochs?.length)} / ${count(run.config?.epochs)} epochs complete</span></div><p class="budget-note">${count(run.action_presentations)} action presentations · ${count(run.train_sessions)} training / ${count(run.validation_sessions)} validation sessions.${latest?`<br>Validation action agreement ${(100*num(latest.action_agreement)).toFixed(1)}%.`:''}<br>Human imitation only; proposals do not override the competitive player. ${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    if(run.schema==='drmc-public-input-alignment-v1'){
      const target=num(run.training_rows)*num(run.config?.epochs);
      const latest=run.epochs?.at(-1)?.validation;
      $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${bar(run.root_presentations,target)}<div class="training-detail"><span>${count(run.root_presentations)} / ${count(target)} example presentations</span><span>${count(run.epochs?.length)} / ${count(run.config?.epochs)} epochs complete</span></div><p class="budget-note">${count(run.training_rows)} training positions · ${count(run.validation_rows)} validation positions.${latest?`<br>Validation policy KL ${Number(latest.kl).toFixed(4)} · ${(100*num(latest.agreement)).toFixed(1)}% choice agreement.`:''}<br>Matches the frozen player's behavior on full public inputs. Playing strength is evaluated separately. ${ago(age(run.updated_at))}</p></div>`);
      continue;
    }
    const source=run.schema==='drmc-public-quality-bank-job-v1';
    const done=source?run.games:run.states, target=source?run.target_games:run.target_states;
    const detail=source?`${count(run.natural_games)} natural games · ${count(run.censored_games)} capped games`:
      `${count(run.rollouts)} terminal continuations · ${count(run.censored_rollouts)} capped continuations`;
    $('#training').insertAdjacentHTML('beforeend',`<div class="training-body"><div class="training-detail"><strong>${esc(run.label)}</strong><span>${esc(run.status)}</span></div>${target?bar(done,target):''}<div class="training-detail"><span>${target?`${count(done)} / ${count(target)} ${source?'source games':'complete positions'}`:'Waiting for the GPU slot'}</span><span>${esc(run.phase || '')}</span></div><p class="budget-note">${target?esc(detail):''}<br>Offline native SMDP data; excluded from training and tournament counts. ${ago(age(run.updated_at))}</p></div>`);
  }
  $('#milestones').innerHTML = (experiment.variants || []).map(v=>{
    const games=(r.tournaments || []).filter(m=>m.a===v.id||m.b===v.id).reduce((n,m)=>n+num(m.played),0);
    const label=ready.has(v.id)?'Ready':v.status==='Training'?'Training':'Pending';
    return `<div class="milestone"><span class="tag ${ready.has(v.id)?'good':'wait'}">${label}</span><strong>${esc(v.name)}</strong><small>${games?`${count(games)} tournament games`:'No completed games yet'}</small></div>`;
  }).join('');
  const errors=[];
  if(feedError)errors.push(feedError);
  if(['failed','stale'].includes(experiment.health?.severity))errors.push(`${experiment.health.status}: ${experiment.health.message}`);
  for(const worker of workers) if(worker.status==='Failed')errors.push(`${worker.host}: ${worker.error || 'Evaluation failed'}`);
  for(const run of experiment.research_runs || []) if(run.status==='Failed')errors.push(`${run.label}: ${run.error || 'Study failed'}`);
  $('#alert').hidden=!errors.length;
  $('#alert').innerHTML=errors.length?`<strong>Attention needed</strong>${errors.map(esc).join('<br>')}`:'';
}

function renderSchedule() {
  const all=experiment?.results?.tournaments || [], mode=$('#match-state').value, query=$('#match-search').value.trim().toLowerCase();
  const matches=all.filter(m=>{
    const complete=num(m.played)>=num(m.target),waiting=m.status==='Waiting for checkpoint';
    return (!query||`${entrantName(m.a)} ${entrantName(m.b)}`.toLowerCase().includes(query)) &&
      (mode==='all'||mode==='complete'&&complete||mode==='waiting'&&waiting||mode==='active'&&!complete&&!waiting);
  }).sort((a,b)=>(a.status==='Playing'?-1:0)-(b.status==='Playing'?-1:0)||num(b.played)-num(a.played));
  $('#schedule-count').textContent=`${count(all.reduce((n,m)=>n+num(m.played),0))} / ${count(all.reduce((n,m)=>n+num(m.target),0))} games`;
  $('#matchups').innerHTML=matches.map(m=>`<tr><td class="matchup-name">${esc(entrantName(m.a))}<br><span>vs ${esc(entrantName(m.b))}</span></td><td>${esc(conditionName(m))}</td><td class="mini-progress"><span class="progress-text">${count(m.played)} / ${count(m.target)}</span>${bar(m.played,m.target)}<small>${esc(m.status || 'Queued')}</small></td><td class="number">${count(m.wins)} / ${count(m.losses)} / ${count(m.draws)}</td><td class="number">${m.censored?`Incomplete · ${count(m.censored)} capped`:m.played?`${(100*(num(m.wins)+.5*num(m.draws))/m.played).toFixed(1)}%`:'—'}</td></tr>`).join('')||'<tr><td colspan="5" class="empty">No matchups in this view.</td></tr>';
}

function render() {
  if(!experiment)return;
  $('#updated').textContent=ago(age(experiment.results?.updated_at));
  renderStandings();renderOperations();renderSchedule();
  $('#study-notes').innerHTML=`<div><h3>Current work</h3><p>${esc(experiment.current_work)}</p><h3 style="margin-top:18px">Study goals</h3><ul>${(experiment.goals||[]).map(g=>`<li>${esc(g)}</li>`).join('')}</ul></div><div><h3>How to read the standings</h3><p>Zero is the selected reference player. Elo differences compare players in this tournament. The intervals are approximate and account for side-swapped seed pairs. A small gap with a wide interval is unresolved.</p><h3 style="margin-top:18px">Evaluation</h3><p>Full controller execution, natural outcomes, and seeds reserved from pace training. Historical cores retain their earlier training exposure. Only evaluation games contribute to these ratings; they are not calibrated to human ratings.</p></div>`;
  renderArchive();
}
$('#conditions').onchange=event=>{condition=event.target.value;save('condition',condition);renderStandings();};
$('#reference').onchange=event=>{reference=event.target.value;save('reference',reference);renderStandings();};
$('#match-state').onchange=renderSchedule;
$('#match-search').oninput=renderSchedule;

let replay=null,replayIndex=0,replayPlaying=false,replayRequest=0,replayRAF=0,replayLoading='';
const NES_FPS=60.0988, colors=['#efcc58','#ea7b76','#78aef1','#8e9daa'];
const replayFrame = i => Number.isFinite(replay.replay[i].frame)?replay.replay[i].frame:i*2;
const clock = frame => {const seconds=Math.floor(frame/NES_FPS);return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;};
function drawReplay() {
  if(!replay?.replay?.length)return;
  const item=replay.replay[replayIndex],canvas=$('#replay-canvas'),ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,820,440);
  for(let side=0;side<2;side++){
    const left=side?518:110,top=30,size=23;
    ctx.fillStyle='#0a1018';ctx.fillRect(left-2,top-4,8*size+4,16*size+8);
    ctx.strokeStyle='#74849b';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,top+16*size);ctx.lineTo(left+8*size,top+16*size);ctx.lineTo(left+8*size,top);ctx.stroke();
    const tile=(x,y,color,virus=false)=>{
      if(y<0||y>=16||x<0||x>=8)return;
      const px=left+x*size,py=top+y*size;
      ctx.fillStyle=colors[color&3];ctx.beginPath();
      if(virus)ctx.arc(px+size/2,py+size/2,9,0,Math.PI*2);else ctx.roundRect(px+1.5,py+1.5,size-3,size-3,5);
      ctx.fill();ctx.fillStyle='#ffffff35';ctx.fillRect(px+6,py+4,8,2);
      if(virus){ctx.fillStyle='#101823';ctx.fillRect(px+6,py+9,3,4);ctx.fillRect(px+14,py+9,3,4);}
    };
    const board=item.boards[side]||[];
    for(let i=0;i<128;i++){const v=board[i]||0;if(!v||v===255||(v&0xf0)===0xf0||(v&0xf0)===0xb0)continue;tile(i%8,Math.floor(i/8),v,(v&0xf0)===0xd0);}
    const pill=item.pills?.[side];
    if(Array.isArray(pill)&&pill.length===5){const[x,y,r,a,b]=pill;if(Number.isInteger(r)&&r>=0&&r<4){const dx=[[0,1],[0,0],[1,0],[0,0]][r],dy=[[0,0],[0,-1],[0,0],[-1,0]][r];[a,b].forEach((c,i)=>tile(x+dx[i],y+dy[i],c));}}
  }
  $('#replay-time').textContent=`${clock(replayFrame(replayIndex))} / ${clock(replayFrame(replay.replay.length-1))}`;
  $('#replay-seek').value=replayIndex;
}
function pauseReplay(){replayPlaying=false;cancelAnimationFrame(replayRAF);$('#replay-toggle').textContent='Play';}
function playReplay(){
  if(!replay?.replay?.length)return;
  if(replayIndex===replay.replay.length-1)replayIndex=0;
  cancelAnimationFrame(replayRAF);replayPlaying=true;$('#replay-toggle').textContent='Pause';
  const start=performance.now(),origin=replayFrame(replayIndex),speed=num($('#replay-speed').value);
  const tick=now=>{const target=origin+(now-start)*NES_FPS*speed/1000;const previous=replayIndex;while(replayIndex+1<replay.replay.length&&replayFrame(replayIndex+1)<=target)replayIndex++;if(previous!==replayIndex)drawReplay();if(replayIndex===replay.replay.length-1)pauseReplay();else replayRAF=requestAnimationFrame(tick);};
  replayRAF=requestAnimationFrame(tick);
}
async function loadReplay(id){
  pauseReplay();const request=++replayRequest;replayLoading=String(id);
  $('#replay-empty').textContent='Loading recorded game…';$('#replay-empty').hidden=false;
  try{
    const response=await fetch(`/api/replay/${encodeURIComponent(id)}`,{signal:AbortSignal.timeout(20000)});if(!response.ok)throw Error(`HTTP ${response.status}`);
    const data=await response.json();if(request!==replayRequest)return;if(!data.replay?.length)throw Error('This game has no recorded frames.');
    replay=data;replay.id=String(id);replayIndex=0;$('#replay-picker').value=replay.id;
    $('#replay-empty').hidden=true;$('#replay-content').hidden=false;
    $('#replay-a').textContent=entrantName(replay.agent_a);$('#replay-b').textContent=entrantName(replay.agent_b);
    $('#replay-result').textContent=replay.winner==='draw'?'Draw':replay.winner==='a'?'A wins':'B wins';
    const match=snapshot?.recorded_matches?.find(m=>String(m.id)===replay.id);
    $('#replay-meta').textContent=`Game ${id} · ${replayCondition(replay)} · seed ${replay.seed} · ${replay.terminal_reason || match?.terminal_reason || 'Recorded game'} · Original A side: ${replay.side_assignment===1?'right':'left'}`;
    $('#replay-seek').max=replay.replay.length-1;drawReplay();
  }catch(error){if(request!==replayRequest)return;$('#replay-empty').textContent=`Could not load replay: ${error.message}`;}
  finally{if(request===replayRequest)replayLoading='';}
}
function renderArchive(){
  const recorded=snapshot?.recorded_matches || [],select=$('#replay-picker');
  const chosen=replayLoading || replay?.id || select.value || String(recorded[0]?.id || '');
  const entries=recorded.map(m=>({value:String(m.id),label:`#${m.id} · ${entrantName(m.agent_a)} vs ${entrantName(m.agent_b)} · ${replayCondition(m)} · seed ${m.seed}`}));
  if(chosen&&!entries.some(e=>e.value===chosen))entries.unshift({value:chosen,label:`#${chosen} · Selected recording`});
  setOptions(select,[{value:'',label:'Select a recorded game'},...entries],chosen);
  if(!replay&&!replayLoading&&chosen)loadReplay(chosen);
}
$('#replay-picker').onchange=event=>{if(event.target.value)loadReplay(event.target.value);};
$('#replay-toggle').onclick=()=>replayPlaying?pauseReplay():playReplay();
$('#replay-speed').onchange=()=>{if(replayPlaying)playReplay();};
function seek(index){if(!replay)return;pauseReplay();replayIndex=Math.max(0,Math.min(replay.replay.length-1,index));drawReplay();}
$('#replay-prev').onclick=()=>seek(replayIndex-1);$('#replay-next').onclick=()=>seek(replayIndex+1);$('#replay-seek').oninput=event=>seek(num(event.target.value));

async function poll(){
  const responses=await Promise.allSettled(['/api/experiment','/api/snapshot'].map(async url=>{const r=await fetch(url,{signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r.json();}));
  if(responses[0].status==='fulfilled')experiment=responses[0].value;
  if(responses[1].status==='fulfilled')snapshot=responses[1].value;
  const failed=responses.some(r=>r.status==='rejected');
  feedError=failed?'A dashboard feed is unavailable. Displayed data may be stale; reconnecting.':'';
  $('#connection').textContent=failed?'Reconnecting':'Connected';$('#connection').classList.toggle('bad',failed);
  try{render();}catch(error){$('#alert').hidden=false;$('#alert').textContent=`Display update failed: ${error.message}`;}
  setTimeout(poll,5000);
}
poll();
