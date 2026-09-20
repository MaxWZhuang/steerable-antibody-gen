"use strict";
const $ = id => document.getElementById(id);
const labels = {continued_sft:"Continued SFT", dpo:"Standard DPO", dpop:"DPOP", dpo_hinge:"DPO + additive hinge", ipo:"IPO", dpo_nll:"DPO + NLL"};
const stageNames = ["SFT controls + DPO β sweep", "DPOP + additive hinge", "IPO + DPO/NLL"];
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite = n => typeof n === "number" && Number.isFinite(n);
const fmt = (n, digits=0) => finite(n) ? n.toLocaleString(undefined, {maximumFractionDigits:digits,minimumFractionDigits:digits}) : "—";
const duration = n => !finite(n) ? "—" : n >= 3600 ? `${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m` : n >= 60 ? `${Math.floor(n/60)}m ${Math.floor(n%60)}s` : `${Math.floor(n)}s`;
const badge = state => `<span class="badge ${esc(state)}">${esc({completed:"Completed",running:"Running",stopped:"Stopped",queued:"Queued",stale:"Heartbeat stale",interrupted:"Interrupted",failed:"Failed",stalled:"No progress"}[state] || state)}</span>`;
const coefficients = r => Object.entries(r.coefficients).map(([k,v]) => `${{beta:"β",lambda:"λ",lambda_site:"λsite",tau:"τ"}[k] || k}=${v}`).join(" · ");
let snapshot = null, selected = null, following = true, details = null, busy = false, requestSerial = 0;

function chart(target, data, field, options={}) {
  const container = $(target);
  const points = data.filter(p => finite(p.x) && finite(p[field]));
  if (!points.length) {container.innerHTML = '<div class="empty-chart">No values recorded for this metric yet.</div>';return;}
  const width=600,height=246,left=49,right=17,top=24,bottom=37;
  const maxX=Math.max(10,...points.map(p=>p.x))*1.03;
  const values=points.map(p=>p[field]);
  if (finite(options.threshold)) values.push(options.threshold,0);
  let lo=Math.min(...values),hi=Math.max(...values);
  const pad=(hi-lo || Math.abs(hi)*.1 || .1)*.12;
  lo-=pad;hi+=pad;
  const x=v=>left+v/maxX*(width-left-right),y=v=>top+(hi-v)/(hi-lo)*(height-top-bottom);
  const tick=v=>Math.abs(v)>0 && Math.abs(v)<.001?v.toExponential(1):Math.abs(v)>999?fmt(v):v.toFixed(2);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.title)}">`;
  for(let i=0;i<5;i++) {const val=lo+(hi-lo)*i/4;svg+=`<line class="grid" x1="${left}" x2="${width-right}" y1="${y(val)}" y2="${y(val)}"/><text x="${left-9}" y="${y(val)+3}" text-anchor="end">${esc(tick(val))}</text>`;}
  for(let i=0;i<5;i++){const val=maxX*i/4;svg+=`<text x="${x(val)}" y="${height-18}" text-anchor="middle">${fmt(val)}</text>`;}
  for(const b of snapshot?.budgets || []) if(b<maxX) svg+=`<line class="budget-line" x1="${x(b)}" x2="${x(b)}" y1="${top}" y2="${height-bottom}"/><text x="${x(b)-4}" y="${top-9}" text-anchor="end">${b}s</text>`;
  if(finite(options.threshold))svg+=`<line class="threshold" x1="${left}" x2="${width-right}" y1="${y(options.threshold)}" y2="${y(options.threshold)}"/><text class="threshold-label" x="${width-right}" y="${y(options.threshold)-7}" text-anchor="end">Stop above ${fmt(options.threshold,1)} nat/seq</text>`;
  svg+=`<polyline class="trace" points="${points.map(p=>`${x(p.x)},${y(p[field])}`).join(" ")}"/>`;
  for(const p of points)svg+=`<circle class="hit" cx="${x(p.x)}" cy="${y(p[field])}" r="5"><title>Update ${p.update} · ${fmt(p.x,2)} GPU s · ${fmt(p[field],5)}</title></circle>`;
  const end=points[points.length-1];
  svg+=`<circle class="last" cx="${x(end.x)}" cy="${y(end[field])}" r="4"/><text x="${(width+left-right)/2}" y="${height-1}" text-anchor="middle">Training GPU seconds (monitoring excluded)</text></svg>`;
  container.innerHTML=svg;
}

function renderOverview() {
  const s=snapshot, counts=s.counts;
  $("campaign-status").outerHTML=`<span id="campaign-status" class="badge ${esc(s.status)}">${esc(s.status==='running'?'Campaign running':s.status==='stale'?'Heartbeat stale':s.status)}</span>`;
  $("phase").textContent=(s.active_phase || "No active phase").replaceAll("_"," · ");
  $("elapsed").textContent=duration(s.elapsed_wall_seconds);
  $("training").textContent=duration(s.recorded_training_gpu_seconds);
  $("allocation").textContent=`of ${duration(s.training_allocation_gpu_seconds)} allocated · actual recorded cost`;
  $("monitoring").textContent=duration(s.recorded_monitor_gpu_seconds);
  const done=(counts.completed||0)+(counts.stopped||0);
  $("finished").textContent=`${done} / ${s.total_runs}`;
  $("counts").textContent=`${counts.completed||0} completed · ${counts.stopped||0} stopped · ${counts.failed||0} failed`;
  $("gpu").textContent=s.gpu?`${fmt(s.gpu.utilization)}%`:"Unavailable";
  $("gpu-memory").textContent=s.gpu?`${fmt(s.gpu.memory_used_mib/1024,1)} / ${fmt(s.gpu.memory_total_mib/1024,1)} GiB · ${s.gpu.name.replace('NVIDIA GeForce ','')}`:"Device telemetry unavailable";
  $("checkpoint").textContent=duration(s.recorded_checkpoint_wall_seconds);
  $("checkpoint-saves").textContent=`${fmt(s.recorded_checkpoint_saves)} rolling saves · one full model write and hash per passing check`;
  $("stages").innerHTML=s.stages.map(st=>`<div class="stage ${(s.active_phase || '').startsWith(`stage${st.stage}_`)?'active':''} ${st.frozen?'done':''}"><b><span class="stage-num">0${st.stage}</span>${stageNames[st.stage-1]}</b><small>${st.counts.completed||0} completed · ${st.counts.stopped||0} stopped · ${st.total} trajectories${st.frozen?' · selection frozen':''}</small></div>`).join("");
  const warnings=[...(s.warnings||[])];
  if(s.status==='stale')warnings.unshift(`The campaign heartbeat is ${duration(s.heartbeat_age_seconds)} old. These are the last saved records; continued execution is unconfirmed.`);
  if(s.error)warnings.unshift(typeof s.error==='string'?s.error:JSON.stringify(s.error));
  $("notice").hidden=!warnings.length;$("notice").textContent=warnings.join("\n");
  $("connection").textContent=s.status==='stale'?"Records may be stale":"Live · 5s refresh";
  $("connection-dot").className=`dot ${s.status==='stale'?'error':'live'}`;
  $("updated").textContent=`Snapshot ${new Date(s.generated_at).toLocaleTimeString()}`;
  const log=$("logs"), atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;
  log.textContent=s.log.length?s.log.join("\n"):"No log lines recorded for this phase yet.";
  if(atBottom)log.scrollTop=log.scrollHeight;
  $("log-label").textContent=s.active_phase || "";
}

function populateSelect() {
  const signature=snapshot.runs.map(r=>`${r.id}:${r.status}`).join("|");
  if($("run-select").dataset.signature!==signature) {
    $("run-select").innerHTML=[1,2,3].map(stage=>`<optgroup label="Stage ${stage}">${snapshot.runs.filter(r=>r.stage===stage).map(r=>`<option value="${esc(r.id)}">${esc(labels[r.objective] || r.objective)} ${esc(coefficients(r))} · ${r.seed} · ${r.status}</option>`).join("")}</optgroup>`).join("");
    $("run-select").dataset.signature=signature;
  }
  if(following && snapshot.active_run)selected=snapshot.active_run;
  if(!selected || !snapshot.runs.some(r=>r.id===selected))selected=snapshot.active_run || snapshot.runs.find(r=>r.status!=='queued')?.id || snapshot.runs[0]?.id;
  $("run-select").value=selected || "";
  $("follow").setAttribute("aria-pressed",String(following));
}

function renderSelected() {
  const r=snapshot.runs.find(r=>r.id===selected);if(!r)return;
  $("run-title").textContent=`${labels[r.objective] || r.objective}${coefficients(r)?' · '+coefficients(r):''}`;
  $("run-meta").innerHTML=`${badge(r.status)}<span>Seed <b>${r.seed}</b></span><span>Updates <b>${fmt(r.updates)}</b></span><span>Training <b>${fmt(r.training_gpu_seconds,1)} / ${Math.max(...snapshot.budgets)} GPU s</b></span><span>Checks <b>${fmt(r.checks)}</b></span><span>Sequence exposures <b>${fmt(r.exposures?.sequences)}</b></span>`;
  const stop=r.stop_reason;
  $("run-note").textContent=r.status==='stalled'?`No artifact written for ${duration(r.idle_seconds)}. The campaign heartbeat is current, so the supervisor is alive; this trajectory's progress is not confirmed.`:r.failure?"Later budgets are not credited. This ended on the artifact failure below, not on a gate verdict, so it does not advance a stage.":(r.status==='stopped'||r.status==='failed')?`Run ended: ${typeof stop==='string'?stop:JSON.stringify(stop || r.status)}. Later budgets are not credited.`:r.gate?`Latest gate: update ${fmt(r.gate.update)} · ${fmt(r.gate.training_gpu_seconds,1)} training GPU seconds · ${fmt(r.gate.pairs)} fixed validation pairs.`:"Waiting for the first completed likelihood check.";
  renderFailure(r);
  $("drop").textContent=fmt(r.gate?.D,3);
  $("gate-label").textContent=r.gate?`${r.gate.passed?'Passing':'Breach'} · threshold ${fmt(snapshot.gate.threshold_nats_per_sequence,1)}`:"No check yet";
  $("chosen-nll").textContent=fmt(r.gate?.mean_current_chosen_nll_per_residue,4);
  $("ranking").textContent=finite(r.gate?.pair_accuracy)?`${fmt(r.gate.pair_accuracy*100,2)}%`:"—";
  const qs=r.gate?.quantiles;
  $("quantiles").innerHTML=qs?[["0.05","p05"],["0.25","p25"],["0.5","p50"],["0.75","p75"],["0.95","p95"]].map(([key,label])=>`<span>${label}<b>${fmt(qs[key],2)}</b></span>`).join(""):"No check recorded";
  // The gate stops on the mean. A passing mean with a collapsed tail and a
  // passing mean with a uniform shift are different states, and only these
  // fractions separate them.
  const fr=r.gate?.fractions;
  $("fractions").innerHTML=fr?[["fraction_below_parent","below parent"],["fraction_drop_gt1","lost &gt;1 nat"],["fraction_drop_gt5","lost &gt;5 nats"],["fraction_below_uniform","below uniform"]].map(([key,label])=>`<span>${label}<b>${finite(fr[key])?fmt(fr[key]*100,2)+"%":"—"}</b></span>`).join(""):"Not recorded for this check";
}

function renderFailure(r) {
  // A gate stop and an artifact failure are different outcomes: one is a
  // result and advances a stage, the other is broken I/O and does not. The
  // trainer separates them carefully and this panel keeps them apart.
  const panel=$("run-failure"), parts=[];
  if(r.failure)parts.push(`<b>${esc(r.failure.type || "Failure")} — not a gate stop</b>${esc(r.failure.message || "")}${r.failure.where?` (${esc(r.failure.where)}, update ${fmt(r.failure.update)})`:""}${r.failure.note?`\n${esc(r.failure.note)}`:""}`);
  for(const e of r.artifact_errors || [])parts.push(`<b>Check ${fmt(e.check)} · artifact write failed</b>${esc(e.save_failed || "")}${e.consequence?`\n${esc(e.consequence)}`:""}`);
  panel.hidden=!parts.length;
  panel.innerHTML=parts.join("<br><br>");
}

function renderTable() {
  const stage=$("stage-filter").value, queued=$("show-queued").checked;
  const rows=snapshot.runs.filter(r=>(stage==='all'||String(r.stage)===stage)&&(queued||r.status!=='queued'));
  $("runs").innerHTML=rows.length?rows.map(r=>`<tr class="${r.id===selected?'selected':''}"><td><button class="run-link" data-run="${esc(r.id)}">${esc(labels[r.objective] || r.objective)}</button><small>Stage ${r.stage}${coefficients(r)?' · '+esc(coefficients(r)):''}</small></td><td>${r.seed}</td><td>${badge(r.status)}</td><td>${r.status==='queued'?'—':fmt(r.updates)}${r.status==='stalled'?`<small>idle ${esc(duration(r.idle_seconds))}</small>`:''}</td><td>${r.status==='queued'?'—':fmt(r.training_gpu_seconds,1)}</td><td>${fmt(r.gate?.D,3)}</td><td>${r.budgets_reached.map(b=>`<span class="budget-chip">${b}</span>`).join("") || '<span class="muted">None</span>'}</td></tr>`).join(""):'<tr><td colspan="7" class="muted">No started trajectories in this stage. Enable “Show queued” to see the declared arms.</td></tr>';
}

function renderPlots() {
  if(!details || details.id!==selected)return;
  chart("likelihood-chart",details.gates.map(p=>({...p,x:p.training_gpu_seconds})),"D",{threshold:snapshot.gate.threshold_nats_per_sequence,title:"Chosen-likelihood drop versus training GPU time"});
  const metric=$("metric").value, label=$("metric").selectedOptions[0].textContent;
  chart("training-chart",details.updates.map(p=>({...p,x:p.cumulative_gpu_seconds})),metric,{title:`${label} versus training GPU time`});
  $("trace-description").textContent=`Recorded training batches · ${label.toLowerCase()}`;
  $("trace-note").textContent=`${fmt(details.update_count)} recorded updates. ${details.plot_stride>1?`First 100 + every ${details.plot_stride}th later update shown; no smoothing.`:'All recorded updates shown; no smoothing.'} Hover a point for its value.`;
}

async function fetchDetail() {
  if(!selected)return;
  const serial=++requestSerial, id=selected;
  const response=await fetch(`/api/run?id=${encodeURIComponent(id)}`,{cache:"no-store",signal:AbortSignal.timeout(15000)});
  if(!response.ok)throw Error(`Trajectory request failed (${response.status})`);
  const result=await response.json();
  if(serial===requestSerial && selected===id){details=result;renderPlots();}
}

async function refresh() {
  if(busy)return;busy=true;
  try {
    const response=await fetch('/api/status',{cache:"no-store",signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw Error(`Snapshot request failed (${response.status})`);
    snapshot=await response.json();renderOverview();populateSelect();renderSelected();renderTable();await fetchDetail();
  } catch(err) {
    $("connection").textContent="Disconnected · retrying";$("connection-dot").className="dot error";
    $("notice").hidden=false;$("notice").textContent=`Live refresh failed. Displayed values may be out of date. ${err.message}`;
  } finally {busy=false;}
}

// ---------------------------------------------------------------------------
// Support-audit view. Additive: it shares the page and touches none of the
// campaign state above. Everything it renders comes from /api/audit, and a value
// the audit did not record is shown as "—" rather than filled in.
// ---------------------------------------------------------------------------
let auditView = false, auditBusy = false;
const pct = n => finite(n) ? `${fmt(n*100,2)}%` : "—";

function renderAudit(a) {
  const status = !a.present ? "No audit run directory yet"
    : a.complete ? "Audit complete"
    : a.active_stage ? `Running · ${a.active_stage}`
    : "Not complete";
  $("audit-status").outerHTML=`<span id="audit-status" class="badge ${a.complete?'completed':a.active_stage?'running':'queued'}">${esc(status)}</span>`;
  $("audit-note").textContent=a.present?(a.checkpoint?`Current: ${a.checkpoint}`:(a.protocol||"")):"Run the inventory stage to create it.";
  const counts=a.counts||{}, coverage=a.coverage||{};
  $("audit-verified").textContent=fmt(counts.states_verified);
  $("audit-enumerated").textContent=`${fmt(counts.states_enumerated)} enumerated · ${fmt(counts.states_expected)} declared`;
  $("audit-scored").textContent=`${fmt(counts.scored)} / ${fmt(counts.distinct_computations)}`;
  $("audit-banks").textContent=fmt(counts.parent_banks);
  $("audit-ches").textContent=`${fmt(counts.ches_parent_blocks)} + ${fmt(counts.ches_endpoint_blocks)}`;
  $("audit-increments").textContent=`${fmt(counts.increments)} early-to-later increments · ${fmt(a.results?.ches_increment_gaps)} reported gaps`;
  $("audit-freeze").textContent=a.frozen?`${String(a.frozen.commit||"").slice(0,10)}`:"Not frozen";
  $("audit-freeze-detail").textContent=a.frozen?`${fmt(a.frozen.sources)} sources · ${fmt(a.frozen.inputs)} inputs · ${fmt(a.frozen.evidence)} evidence files`:"score and ches refuse to run without the marker";
  const immutable=a.verification?.immutable;
  $("audit-immutable").textContent=immutable===true?"Verified":immutable===false?"Problems":"—";
  $("audit-immutable-detail").textContent=`${fmt(a.verification?.shards_checked)} shards checked${(a.verification?.problems||[]).length?` · ${a.verification.problems.length} problem(s)`:""}`;
  $("audit-stages").innerHTML=(a.stages||[]).map(s=>{
    const progress=finite(s.fraction)?`${fmt(s.fraction*100,0)}%`:(finite(s.completed)?`${fmt(s.completed)} done`:"—");
    const total=finite(s.total)?` / ${fmt(s.total)}`:" · total not knowable in advance";
    return `<div class="stage ${s.status==='running'?'active':''} ${s.status==='completed'?'done':''}"><b>${esc(s.stage)} ${badge(s.status==='not_started'?'queued':s.status)}</b><small>${esc(progress)}${esc(total)}${s.error?` · ${esc(s.error)}`:""}${s.stale?" · heartbeat stale":""}</small></div>`;}).join("");
  const shortfalls=(coverage.shortfalls||[]);
  $("audit-coverage-label").textContent=coverage.complete===true?"complete":coverage.complete===false?"gaps reported":"not evaluated";
  $("audit-coverage").innerHTML=shortfalls.length
    ? shortfalls.map(s=>`<span>${esc(s.role)} · ${esc(s.kind||"enumeration")} <b>${fmt(s.observed)} / ${fmt(s.expected)}</b></span>`).join("")
    : (coverage.complete?`<span>Every declared population was located <b>and verified</b></span>`:`<span class="muted">Coverage has not been evaluated yet</span>`);
  const rows=(a.results?.endpoints)||[];
  $("audit-results-label").textContent=`${fmt(rows.length)} scored`;
  $("audit-endpoints").innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(r.id)}<small>${esc(r.role||"")}</small></td><td>${esc(r.arm_id||"—")}</td><td>${esc(r.seed??"—")}</td><td>${finite(r.budget_gpu_seconds)?fmt(r.budget_gpu_seconds):"—"}</td><td>${fmt(r.forward_kl,4)}</td><td>${finite(r.ci_low)?`[${fmt(r.ci_low,4)}, ${fmt(r.ci_high,4)}]`:"—"}</td><td>${pct(r.tenfold_fraction)}</td><td>${pct(r.tenfold_wilson_lower)}</td></tr>`).join(""):'<tr><td colspan="8" class="muted">No scored computations yet. Scoring refuses to run before the freeze.</td></tr>';
  const decision=a.decision;
  $("audit-outcome").textContent=decision?decision.outcome.replaceAll("_"," "):"Not decided";
  $("audit-methods").innerHTML=decision?Object.entries(decision.methods||{}).map(([name,m])=>`<tr><td>${esc(name)}</td><td>${fmt(m.seeds_usable)} / ${fmt(m.seeds_declared)}</td><td>${fmt(m.seeds_crossing)}</td><td>${esc(m.outcome)}</td></tr>`).join(""):'<tr><td colspan="4" class="muted">The decision is a function of the scored artifacts.</td></tr>';
  const blocking=[...(decision?.blocking||[]),...(a.unmet_requirements||[]).map(r=>`unmet requirement: ${r}`)];
  $("audit-blocking").textContent=blocking.length?`This audit cannot support a preservation finding yet — ${blocking.join("; ")}.`:(a.complete?"Every declared requirement was satisfied and verified.":"");
  const problems=[...(a.errors||[]),...(a.verification?.problems||[]).map(p=>p.problem||JSON.stringify(p))];
  $("audit-notice").hidden=!problems.length;$("audit-notice").textContent=problems.join("\n");
  $("audit-updated").textContent=`Snapshot ${new Date(a.generated_at).toLocaleTimeString()}`;
}

async function refreshAudit() {
  if(auditBusy)return;auditBusy=true;
  try {
    const response=await fetch('/api/audit',{cache:"no-store",signal:AbortSignal.timeout(15000)});
    if(response.status===404){renderAudit({present:false,stages:[],generated_at:new Date().toISOString()});return;}
    if(!response.ok)throw Error(`Audit request failed (${response.status})`);
    renderAudit(await response.json());
  } catch(err) {
    $("audit-notice").hidden=false;$("audit-notice").textContent=`Audit refresh failed. Displayed values may be out of date. ${err.message}`;
  } finally {auditBusy=false;}
}

// ---------------------------------------------------------------------------
// Parent-replay view. Additive again: everything it renders comes from
// /api/replay, every value is written with textContent rather than innerHTML for
// anything that came out of a run directory, and a value the campaign did not
// record is shown as "—" rather than filled in. It never says a launch is healthy
// without an actual completed update and an actual passed check.
// ---------------------------------------------------------------------------
let replayView = false, replayBusy = false;
const replayStatusLabel = {queued:"Queued", running:"Running", completed:"Completed",
  stopped_by_gate:"Stopped by gate", incomplete:"Incomplete", failed:"Failed",
  interrupted:"Interrupted", unknown:"Unknown"};
const phaseLabel = {no_run_directory:"No run directory", before_freeze:"Before the freeze",
  preparing_banks:"Preparing banks", banks_ready:"Banks ready", fitting:"Fitting"};
// The launch tile is never an unqualified "Healthy". The first update and the
// first passed check stay true after the writer dies, so the label carries what
// the campaign is doing now and the evidence stays in the detail line.
const healthLabel = {not_established:"Not established", healthy:"Healthy · running",
  launch_verified:"Launch verified", launch_verified_but_interrupted:"Launched · interrupted",
  launch_verified_but_campaign_failed:"Launched · campaign failed",
  launch_verified_and_session_finished:"Launched · session finished"};
const healthBadge = {not_established:"queued", healthy:"running",
  launch_verified:"queued", launch_verified_but_interrupted:"interrupted",
  launch_verified_but_campaign_failed:"failed",
  launch_verified_and_session_finished:"completed"};

function cell(row, text, title) {
  const td = document.createElement("td");
  td.textContent = text;
  if(title){const small=document.createElement("small");small.textContent=title;td.appendChild(small);}
  row.appendChild(td);
  return td;
}

function renderReplay(r) {
  const phase = phaseLabel[r.phase] || r.phase || "—";
  const state = !r.present ? "No replay run directory yet"
    : r.writer_state === "failed" ? "Campaign failed"
    : r.writer_state === "gone" ? "Writer gone · lock released"
    : r.writer_state === "unknown" ? "Writer state unknown"
    : r.writer_state === "alive" ? `Running · ${phase}`
    : r.writer_state === "finished" ? "Session finished"
    : phase;
  const badgeClass = !r.present ? "queued"
    : r.writer_state === "failed" ? "failed"
    : r.writer_state === "gone" ? "interrupted"
    : r.writer_state === "unknown" ? "interrupted"
    : r.writer_state === "alive" ? "running"
    : r.writer_state === "finished" ? "completed" : "queued";
  $("replay-status").outerHTML=`<span id="replay-status" class="badge ${badgeClass}">${esc(state)}</span>`;
  $("replay-note").textContent = r.present
    ? (r.queue_present ? `${fmt(r.total)} declared trajectories · ${esc(r.campaign_id||"")}`
                       : "The campaign has not been launched; there is no queue yet.")
    : "Run the prepare stage to create it.";
  const health = r.health || {};
  const lockState = (r.lock && r.lock.state) || "not observed";
  $("replay-health").textContent = healthLabel[health.status] || (health.healthy ? "Launch verified" : "Not established");
  $("replay-health").dataset.state = healthBadge[health.status] || "queued";
  $("replay-health-detail").textContent = health.healthy
    ? `first update ${fmt(health.first_update_completed)} · first check at update ${fmt(health.first_check_update)} passed · lock ${esc(lockState)}${finite(r.heartbeat_age_seconds) ? ` · heartbeat ${fmt(Math.round(r.heartbeat_age_seconds))}s old` : ""}`
    : (health.detail || health.basis || "Needs a completed update and a passed check");
  const counts = r.counts || {};
  const done = (counts.completed||0)+(counts.stopped_by_gate||0);
  $("replay-finished").textContent = r.queue_present ? `${done} / ${fmt(r.total)}` : "—";
  $("replay-counts").textContent = r.queue_present
    ? `${counts.completed||0} completed · ${counts.stopped_by_gate||0} stopped · ${counts.incomplete||0} incomplete · ${counts.queued||0} queued`
    : "No queue document yet";
  $("replay-active").textContent = r.active || "—";
  $("replay-active-detail").textContent = r.heartbeat && r.heartbeat.update
    ? `update ${fmt(r.heartbeat.update)} · ${fmt(r.heartbeat.exposures?.chosen)} chosen exposures`
    : "No trajectory is writing right now";
  $("replay-freeze").textContent = r.frozen ? String(r.frozen.commit||"").slice(0,10) : "Not frozen";
  $("replay-freeze-detail").textContent = r.frozen
    ? `${fmt(r.frozen.sources)} sources · ${fmt(r.frozen.inputs)} inputs · audit ${esc(r.frozen.audit_decision||"—")}`
    : "fitting refuses to start without the marker";
  $("replay-banks").textContent = r.banks ? `${fmt(r.banks.count)} artifacts` : "Not generated";
  $("replay-banks-detail").textContent = r.banks
    ? `bound to freeze ${String(r.banks.freeze_commit||"").slice(0,10)}`
    : "generated after the freeze and bound to it";
  const results = r.results || {};
  $("replay-endpoints").textContent = finite(results.reached_endpoint_rows)
    ? `${fmt(results.reached_endpoint_rows)} / ${fmt(results.declared_endpoint_rows)}` : "—";
  $("replay-deltas").textContent = finite(results.matched_deltas)
    ? `${fmt(results.matched_deltas)} matched · ${fmt(results.unavailable_deltas)} unavailable`
    : "No report has been built yet";
  const body = $("replay-rows");
  body.textContent = "";
  const rows = r.trajectories || [];
  $("replay-queue-label").textContent = rows.length ? `${fmt(rows.length)} declared` : "";
  if(!rows.length){
    const tr=document.createElement("tr");const td=document.createElement("td");
    td.colSpan=10;td.className="muted";
    td.textContent=r.present?"No queue document yet: the campaign has not been launched.":"No replay run directory yet.";
    tr.appendChild(td);body.appendChild(tr);
  }
  for(const row of rows){
    const tr=document.createElement("tr");
    cell(tr,row.trajectory||"—",row.is_control?"matched zero-replay control":"");
    cell(tr,labels[row.task]||row.task||"—");
    cell(tr,finite(row.replay_lambda)?String(row.replay_lambda):"—");
    cell(tr,row.seed??"—");
    const status=document.createElement("td");
    status.innerHTML=`<span class="badge ${esc(row.status)}">${esc(replayStatusLabel[row.status]||row.status)}</span>`;
    if(row.stop_reason){const small=document.createElement("small");small.textContent=row.stop_reason;status.appendChild(small);}
    tr.appendChild(status);
    cell(tr,finite(row.updates)?fmt(row.updates):"—");
    cell(tr,finite(row.chosen_exposures)?fmt(row.chosen_exposures):"—",
         finite(row.replay_exposures)&&row.replay_exposures?`${fmt(row.replay_exposures)} replay rows`:"");
    cell(tr,finite(row.gate?.D)?fmt(row.gate.D,4):"—",row.gate&&row.gate.passed===false?"gate breached":"");
    cell(tr,finite(row.gate?.forward_kl)?fmt(row.gate.forward_kl,4):"—",
         finite(row.gate?.tenfold_fraction)?`>ln10 ${pct(row.gate.tenfold_fraction)}`:"");
    cell(tr,(row.endpoints_reached||[]).join(", ")||"—");
    body.appendChild(tr);
  }
  const problems=[...(r.warnings||[]),...(r.errors||[])];
  if(r.error)problems.unshift(typeof r.error==="string"?r.error:JSON.stringify(r.error));
  $("replay-notice").hidden=!problems.length;$("replay-notice").textContent=problems.join("\n");
  $("replay-updated").textContent=`Snapshot ${new Date(r.generated_at).toLocaleTimeString()}`;
}

async function refreshReplay() {
  if(replayBusy)return;replayBusy=true;
  try {
    const response=await fetch('/api/replay',{cache:"no-store",signal:AbortSignal.timeout(15000)});
    if(response.status===404){renderReplay({present:false,trajectories:[],generated_at:new Date().toISOString()});return;}
    if(!response.ok)throw Error(`Replay request failed (${response.status})`);
    renderReplay(await response.json());
  } catch(err) {
    $("replay-notice").hidden=false;$("replay-notice").textContent=`Replay refresh failed. Displayed values may be out of date. ${err.message}`;
  } finally {replayBusy=false;}
}

function showView(view) {
  auditView = view === "audit";
  replayView = view === "replay";
  $("campaign-heading").hidden = auditView || replayView;
  $("view-campaign").hidden = auditView || replayView;
  $("view-audit").hidden = !auditView;
  $("view-replay").hidden = !replayView;
  $("tab-campaign").setAttribute("aria-pressed", String(!auditView && !replayView));
  $("tab-audit").setAttribute("aria-pressed", String(auditView));
  $("tab-replay").setAttribute("aria-pressed", String(replayView));
  if(auditView)refreshAudit();
  if(replayView)refreshReplay();
}

function selectRun(id){following=false;selected=id;details=null;populateSelect();renderSelected();renderTable();for(const el of ['likelihood-chart','training-chart'])$(el).innerHTML='<div class="empty-chart">Loading trajectory…</div>';fetchDetail().catch(err=>{$('notice').hidden=false;$('notice').textContent=err.message;});}
// The header button refreshes whatever is on screen. Wiring it to the campaign
// unconditionally meant pressing Refresh on the audit tab reloaded the hidden view.
$("refresh").addEventListener('click',()=>{replayView?refreshReplay():auditView?refreshAudit():refresh();});
$("run-select").addEventListener('change',e=>selectRun(e.target.value));
$("follow").addEventListener('click',()=>{following=!following;populateSelect();renderSelected();renderTable();fetchDetail().catch(()=>{});});
$("metric").addEventListener('change',renderPlots);
$("stage-filter").addEventListener('change',()=>snapshot&&renderTable());
$("show-queued").addEventListener('change',()=>snapshot&&renderTable());
$("runs").addEventListener('click',e=>{const button=e.target.closest('[data-run]');if(button)selectRun(button.dataset.run);});
for(const tab of document.querySelectorAll('[data-view]'))tab.addEventListener('click',()=>showView(tab.dataset.view));
refresh();setInterval(refresh,5000);
setInterval(()=>{if(auditView)refreshAudit();},5000);
setInterval(()=>{if(replayView)refreshReplay();},5000);
