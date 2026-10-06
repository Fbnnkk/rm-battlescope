/* Uses the embedded replay payload; no network or CDN required for playback. */
(() => {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const formatTime = t => `${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
  const num = n => Number(n || 0).toLocaleString('zh-CN', {maximumFractionDigits:1});
  const review = data.review || {};
  const scores = data.scores || [];
  const delivery = data.delivery || {mode:'static'};
  const main = document.querySelector('main');
  const replay = document.createElement('section');
  replay.id = 'bsReplay';
  while (main.firstChild) replay.append(main.firstChild);
  const header = document.createElement('header');
  header.className = 'bs-top';
  header.innerHTML = `<div><div class="bs-brand">RM BATTLESCOPE / MATCH REPORT</div><h1>${esc(data.match.红方学校)} <span class="bs-muted">vs</span> ${esc(data.match.蓝方学校)}</h1><p>${esc(data.match.赛区)} · 第${esc(data.match.场次号)}场 / 第${esc(data.match.局号)}局 · ${data.match._is_partial?'时间窗表现':'整局战绩'} · ${esc(data.match.game_id)}</p></div><span class="bs-kicker">${data.match._is_partial?'片段复盘':esc(data.match.胜方 || '未知')+'方获胜'}</span>`;
  main.append(header);
  const tabs = document.createElement('nav');
  tabs.className = 'bs-tabs'; tabs.setAttribute('aria-label','复盘视图');
  tabs.innerHTML = '<button id="bsOverviewTab" type="button" aria-selected="true">战绩总览</button><button id="bsReplayTab" type="button" aria-selected="false">战术回放</button><span class="spacer"></span><button id="bsExport" type="button">导出战绩 JSON</button>';
  if(delivery.csv_url) tabs.insertAdjacentHTML('beforeend',`<a class="bs-link" href="${esc(delivery.csv_url)}" download>评分 CSV</a>`);
  if(delivery.return_url) tabs.insertAdjacentHTML('beforeend',`<a class="bs-link" href="${esc(delivery.return_url)}">← ${delivery.mode==='backend'?'比赛库':'演示目录'}</a>`);
  main.append(tabs);
  const exportStatus=document.createElement('div');exportStatus.className='bs-notice';exportStatus.setAttribute('aria-live','polite');main.append(exportStatus);
  const overview = document.createElement('section'); overview.id='bsOverview';
  main.append(overview, replay);
  if(location.protocol === 'file:') replay.querySelector('.return-button')?.remove();
  function switchView(name) {
    const overviewActive = name === 'overview'; overview.hidden=!overviewActive; replay.hidden=overviewActive;
    document.getElementById('bsOverviewTab').setAttribute('aria-selected',String(overviewActive));
    document.getElementById('bsReplayTab').setAttribute('aria-selected',String(!overviewActive));
    if(!overviewActive) { resize(); draw(); }
  }
  function seek(t, context=null) { switchView('replay'); stop(); slider.value=Math.max(minT,Math.min(maxT,t)); draw(); document.getElementById('bsSeekContext').textContent=context||''; document.getElementById('bsReplayToolbar').scrollIntoView({block:'start',behavior:'instant'}); }
  document.getElementById('bsOverviewTab').onclick=()=>switchView('overview');
  document.getElementById('bsReplayTab').onclick=()=>switchView('replay');
  const cvalue=(s,k)=>s.components?.[k]?.value || 0;
  const allDamage=s=>['damage_to_robots','damage_to_base','damage_to_outpost'].reduce((n,k)=>n+cvalue(s,k),0);
  const rated=s=>s.rating_eligible!==false && Number.isFinite(s.total_score);
  const scoreText=s=>rated(s)?s.total_score.toFixed(2):'—';
  const evidenceLabel=c=>({recorded:'原始事件记录',observed:'原始遥测',high:'高可信归因',medium:'中可信归因',low:'低可信分摊',inferred:'战术迹象推断',role_inferred:'事件记录 · 执行者推断'}[c]||'未标注来源');
  const best=review.best || [...scores].filter(rated).sort((a,b)=>b.total_score-a.total_score || (b.raw_score||0)-(a.raw_score||0))[0];
  const teams=review.teams || ['红','蓝'].map(camp=>({camp,school:data.match[camp+'方学校']}));
  function board(camp) {
    const members=scores.filter(s=>s.camp===camp).sort((a,b)=>Number(rated(b))-Number(rated(a)) || (b.total_score||0)-(a.total_score||0));
    return `<section class="bs-panel bs-board"><div class="bs-panel-head"><h2 style="color:${colors[camp]}">${esc(data.match[camp+'方学校'])}</h2><span class="bs-muted">兵种 · 击杀/助攻/阵亡 · 输出 · 评分</span></div>${members.map((s,i)=>`<button type="button" class="bs-row" data-focus="${esc(camp)}:${s.robot_id}"><span class="bs-rank">${rated(s)?i+1:'—'}</span><span><strong>${esc(s.robot_type)}</strong><small>${s.event_only?'事件记录':'#'+s.robot_id} · ${s.explanation?.find(c=>c.score>0)?.label ? esc(s.explanation.find(c=>c.score>0).label):esc(s.assessment_status||'证据不足')}</small></span><span>${cvalue(s,'kills')}/${cvalue(s,'assists')}/${cvalue(s,'deaths')}<small>K / A / D</small></span><span>${num(allDamage(s))}<small>${s.evidence?.recorded_damage?'原始命中事件伤害':s.evidence?.low_confidence_damage?'估计伤害 · 含低可信':'高/中可信归因伤害'}</small></span><span class="bs-number">${scoreText(s)}<small class="bs-grade ${rated(s)?esc(s.grade):'unrated'}">${rated(s)?esc(s.grade):'暂不评级'}</small></span></button>`).join('') || '<div class="bs-empty">当前未启用评分</div>'}</section>`;
  }
  overview.innerHTML = `<div class="bs-overview"><div class="bs-best"><div class="bs-kicker">本局最高表现分 ${data.match._is_partial?'· 片段':''}</div>${best?`<div class="bs-best-name"><strong>${esc(best.school)}<div class="bs-muted">${esc(best.camp)}方 · ${esc(best.robot_type)}</div></strong><span class="bs-best-score">${best.total_score.toFixed(2)}</span></div><span class="bs-muted">兵种职责不同，请结合分项与战术背景评价。</span>`:'<div class="bs-empty">尚无评分数据</div>'}</div>${teams.map(t=>`<section class="bs-panel bs-summary ${t.camp==='红'?'red':'blue'}"><h2>${esc(t.school)}</h2><div class="bs-summary-stats"><span><strong>${t.mean_score?.toFixed(1) ?? '—'}</strong><span class="bs-muted">已评级均分</span></span><span><strong>${num(t.kills)}</strong><span class="bs-muted">归因击杀</span></span><span><strong>${num(t.structure_damage)}</strong><span class="bs-muted">建筑估计输出</span></span></div><p class="bs-muted">均分分母 ${num(t.rated_count)} 个已评级实体 · ${num(t.unrated_count)} 个暂不评级</p></section>`).join('')}</div><div class="bs-boards">${board('红')}${board('蓝')}</div><section id="bsFocus" class="bs-panel"></section><details class="bs-panel bs-method"><summary>评分怎么算 · ${esc(review.scoring?.rule_version || '自定义表现分')}</summary><ul>${(review.notes||['表现分由自定义加减分规则生成。']).map(n=>`<li>${esc(n)}</li>`).join('')}<li>计分证据：${review.scoring?.min_confidence==='high'?'仅高可信攻击':review.scoring?.min_confidence==='low'?'含低可信伤害分摊':'高/中可信攻击'}；未计入攻击 ${num(review.scoring?.excluded_attack_count)} 条。${review.scoring?.projectile_attribution_coverage!=null?`原始弹丸受击伤害归因覆盖 ${num(review.scoring.projectile_attribution_coverage*100)}%。`:""}队伍金币消耗不归因到英雄。</li><li>职责分 = 5 + 递减折算后的贡献 − 阵亡代价。每个正向分项共同分摊5分预算；分项和与最终分一致。旧规则分保留供对照。</li></ul></details>`;
  const damageSummary=review.scoring;
  if(damageSummary?.damage_targets){
    const audit=document.createElement('details');audit.className='bs-panel bs-damage-audit';
    audit.innerHTML=`<summary>伤害核对 · 原始受击 ${num(damageSummary.observed_projectile_damage)} HP · 未归因 ${num(damageSummary.unattributed_projectile_damage)} HP <span class="bs-muted">展开分项与建筑明细</span></summary><div class="bs-audit-body"><p>原始受击HP记录与攻击来源推断分开。高、中可信也属于启发式归因，非官方确认。</p><p class="bs-muted">高可信归因 ${num(damageSummary.high_confidence_projectile_damage)} + 中可信归因 ${num(damageSummary.medium_confidence_projectile_damage)} + 低可信分摊 ${num(damageSummary.low_confidence_projectile_damage)} + 未归因 ${num(damageSummary.unattributed_projectile_damage)} = ${num(damageSummary.observed_projectile_damage)} HP</p>${damageSummary.damage_targets.filter(t=>['基地','前哨站'].includes(t.robot_type)).map(t=>`<p><strong>${esc(t.camp)}方${esc(t.robot_type)} · 原始弹丸受击 ${num(t.observed_damage)} HP</strong><br><span class="bs-muted">高可信归因 ${num(t.high_confidence_damage)} + 中可信归因 ${num(t.medium_confidence_damage)} + 低可信分摊 ${num(t.low_confidence_damage)} + 未归因 ${num(t.unattributed_damage)} = ${num(t.observed_damage)}；多人共同分摊一份伤害。飞镖原始命中伤害另计。</span></p>`).join('')}</div>`;
    overview.querySelector('.bs-boards').after(audit);
  }
  let focused=null;
  function focus(key) {
    const s=scores.find(s=>`${s.camp}:${s.robot_id}`===key); if(!s)return; focused=s;
    overview.querySelectorAll('[data-focus]').forEach(el=>el.classList.toggle('active',el.dataset.focus===key));
    const explanations=s.explanation||[];
    const maxDelta=Math.max(.1,...explanations.map(c=>Math.abs(c.score)));
    const contributions=explanations.filter(c=>c.score!==0);
    const e=s.evidence || {};
    const ts=(data.timeseries_scores?.[s.camp]||[]).find(t=>t.robot_id===s.robot_id);
    const width=540,height=210,left=32,right=525,top=14,bottom=182;
    const x=t=>left+(t-minT)/Math.max(1,maxT-minT)*(right-left), y=v=>bottom-v/10*(bottom-top);
    const path=ts?ts.times.map((t,i)=>ts.score[i]==null?'':`${i===0||ts.score[i-1]==null?'M':'L'}${x(t).toFixed(1)},${y(ts.score[i]).toFixed(1)}`).join(' '):'';
    const plot=ts&&rated(s)?`<svg id="bsChart" class="bs-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(s.robot_type)}整局评分曲线，点击跳转回放">${[0,5,10].map(v=>`<line class="${v===5?'baseline':''}" x1="${left}" y1="${y(v)}" x2="${right}" y2="${y(v)}"/><text x="10" y="${y(v)+4}">${v}</text>`).join('')}<path d="${path}"/>${[minT,(minT+maxT)/2,maxT].map(t=>`<text x="${x(t)}" y="204" text-anchor="middle">${formatTime(t)}</text>`).join('')}</svg>`:'<div class="bs-empty">证据不足，暂不展示表现曲线</div>';
    document.getElementById('bsFocus').innerHTML=`<div class="bs-panel-head"><div class="bs-detail-title"><span class="bs-detail-score">${scoreText(s)}</span><div><h2>${esc(s.camp)}方 ${esc(s.robot_type)} <span class="bs-grade ${rated(s)?esc(s.grade):'unrated'}">${rated(s)?esc(s.grade):'暂不评级'}</span></h2><span class="bs-muted">${esc(s.school)} · ${s.event_only?'事件记录':'#'+s.robot_id} · ${esc(s.mission||'')}<br>${s.legacy_score!=null?'旧规则 '+s.legacy_score.toFixed(2)+' → 职责分 '+s.total_score.toFixed(2):esc(s.assessment_status||'')}</span></div></div><button id="bsWatchUnit" type="button">查看回放 →</button></div><div class="bs-detail-grid"><div><div class="bs-muted">${rated(s)?'得分构成 · 中性起点5分':'证据不足 · 暂不评级，不计入队伍均分'}</div>${contributions.map(c=>`<div class="bs-contrib ${c.score<0?'negative':''}"><span>${esc(c.label)} <span class="bs-muted">${typeof c.value==='number'?num(c.value):''}</span></span><span class="bs-contrib-value">${c.score>0?'+':''}${c.score.toFixed(3)}</span><div class="bs-contrib-track"><div class="bs-contrib-fill" style="width:${Math.abs(c.score)/maxDelta*100}%"></div></div></div>`).join('')||`<div class="bs-empty">${esc(s.assessment_status)}<br>${esc(s.assessment_reason)}</div>`}<div class="bs-evidence">${e.mode==='event_only'?'事件记录评分 · 无移动遥测':`血量遥测覆盖 ${num((e.health_coverage||0)*100)}% · 清洗后位置可用 ${num((e.position_coverage||0)*100)}%`}<br>计分攻击 ${num(e.scored_attacks)} / 推断攻击 ${num(e.inferred_attacks)} · 其中高可信 ${num(e.high_confidence_attacks)}<br>高可信归因 ${num(e.high_confidence_damage)} HP · 中可信归因 ${num(e.medium_confidence_damage)} HP · 原始命中事件 ${num(e.recorded_damage)} HP · 低可信分摊 ${num(e.low_confidence_damage)} HP（${num(e.low_confidence_damage_windows)}个受击窗口）<br>低可信仅折算输出分；覆盖率描述数据完整度，不是评分准确率。</div></div><div><div class="bs-muted">整局表现曲线 · 点击曲线跳到相应时刻</div>${plot}<div class="bs-focus-stats"><div class="bs-focus-stat"><strong>${cvalue(s,'kills')} / ${cvalue(s,'assists')} / ${cvalue(s,'deaths')}</strong><span class="bs-muted">击杀 / 助攻 / 阵亡</span></div><div class="bs-focus-stat"><strong>${num(cvalue(s,'damage_to_robots'))}</strong><span class="bs-muted">机器人估计伤害</span></div><div class="bs-focus-stat"><strong>${num(cvalue(s,'damage_to_base')+cvalue(s,'damage_to_outpost'))}</strong><span class="bs-muted">建筑估计伤害</span></div></div><div class="bs-muted" style="margin-top:15px">${(e.limitations||['部分战术贡献未被完整观测，请结合回放判断。']).map(esc).join('<br>')}</div></div></div>`;
    const focusPanel=document.getElementById('bsFocus');
    if(s.dimensions){
      const dimensions=document.createElement('div');dimensions.className='bs-dimensions';
      dimensions.innerHTML=s.dimensions.map(d=>`<div><span>${esc(d.label)}</span><strong class="${d.score<0?'cost':''}">${d.score>0?'+':''}${d.score.toFixed(2)}</strong></div>`).join('');
      focusPanel.querySelector('.bs-detail-grid').before(dimensions);
    }
    if(s.contribution_events){
      const proof=document.createElement('section');proof.className='bs-proof';
      const eventKeys=[...new Map(s.contribution_events.map(i=>[i.key,i.label])).entries()];
      proof.innerHTML=`<div class="bs-panel-head"><h3>贡献依据 · 点击回看</h3><select id="bsProofKind" aria-label="贡献依据类别"><option value="">全部贡献</option>${eventKeys.map(([k,label])=>`<option value="${esc(k)}">${esc(label)}</option>`).join('')}</select></div><div id="bsProofEvents" class="bs-moment-list"></div><button id="bsProofMore" type="button" hidden>显示更多依据</button><div class="bs-muted">标记为推断的贡献只表示观测迹象。压制在后续证据确认时计分，点击从交战起点回看。</div>`;
      focusPanel.append(proof);
      let proofLimit=50;
      function showProof(){
        const selected=document.getElementById('bsProofKind').value;
        const items=s.contribution_events.filter(i=>!selected||i.key===selected).slice().reverse();
        document.getElementById('bsProofMore').hidden=items.length<=proofLimit;
        document.getElementById('bsProofEvents').innerHTML=items.slice(0,proofLimit).map(i=>`<button type="button" class="bs-moment" data-proof-time="${i.start??i.time}"><time>${formatTime(i.time)}</time><span><strong>${esc(i.label)} · ${esc(i.target)} 〔${esc(evidenceLabel(i.confidence))}〕</strong><small>${esc(i.reason)} · 观测量 ${num(i.value)}</small></span></button>`).join('')||'<div class="bs-empty">当前没有可计分贡献记录。中性分不等于零贡献。</div>';
      }
      document.getElementById('bsProofKind').onchange=()=>{proofLimit=50;showProof();};
      document.getElementById('bsProofMore').onclick=()=>{proofLimit+=50;showProof();};
      document.getElementById('bsProofEvents').onclick=event=>{const el=event.target.closest('[data-proof-time]');if(el)seek(Number(el.dataset.proofTime));};
      showProof();
    }
    document.getElementById('bsWatchUnit').onclick=()=>seek(Number(slider.value));
    document.getElementById('bsChart')?.addEventListener('click',event=>{const rect=event.currentTarget.getBoundingClientRect(); const fraction=((event.clientX-rect.left)/rect.width*width-left)/(right-left); seek(minT+fraction*(maxT-minT));});
  }
  overview.addEventListener('click',event=>{const row=event.target.closest('[data-focus]'); if(row)focus(row.dataset.focus);});
  if(best)focus(`${best.camp}:${best.robot_id}`);else if(scores.length)focus(`${scores[0].camp}:${scores[0].robot_id}`);
  // Time navigation and bookmarks sit beside the map, using the same clock.
  const timebar=document.createElement('div');timebar.className='bs-timebar';
  timebar.innerHTML='<label class="bs-muted">跳到 <input id="bsJump" type="number" min="0" step="1" aria-label="跳转秒数"> 秒</label><button id="bsJumpButton" type="button">跳转</button><button id="bsShare" type="button">复制时刻链接</button><span id="bsShareStatus" class="bs-muted" aria-live="polite"></span>';
  document.getElementById('bsSettingsBody').append(timebar);
  document.getElementById('bsPrev').onclick=()=>seek(Number(slider.value)-5);
  document.getElementById('bsNext').onclick=()=>seek(Number(slider.value)+5);
  document.getElementById('bsJumpButton').onclick=()=>{const v=document.getElementById('bsJump').value;if(v!==''&&Number.isFinite(Number(v)))seek(Number(v));};
  document.getElementById('bsShare').onclick=async()=>{const url=new URL(location.href);url.hash=`t=${Number(slider.value).toFixed(2)}`;try{await navigator.clipboard.writeText(url.href);document.getElementById('bsShareStatus').textContent='已复制';}catch{document.getElementById('bsShareStatus').textContent='时刻已写入地址栏，可手动复制';} history.replaceState(null,'',url);};
  const lower=document.createElement('div'); lower.className='bs-review-grid';
  lower.innerHTML='<section class="bs-panel"><div class="bs-panel-head"><h2>关键事件</h2><span id="bsMomentCount" class="bs-muted"></span></div><div class="bs-moment-tools"><select id="bsMomentKind" aria-label="事件类别"><option value="">全部类别</option></select><select id="bsMomentCamp" aria-label="事件阵营"><option value="">双方</option><option>红</option><option>蓝</option></select><input id="bsMomentSearch" type="search" placeholder="搜索兵种或事件" aria-label="搜索关键事件"></div><div id="bsMoments" class="bs-moment-list"></div></section><section class="bs-panel"><div class="bs-panel-head"><h2>复盘笔记</h2><span class="bs-muted">保存在此浏览器</span></div><div class="bs-note-body"><textarea id="bsNoteText" maxlength="2000" placeholder="这一波发生了什么？下次如何调整？" aria-label="复盘笔记内容"></textarea><div class="bs-note-actions"><button id="bsAddNote" type="button">记录当前时刻</button><button id="bsExportNotes" type="button">导出笔记</button><button id="bsImportNotes" type="button">导入笔记</button><input id="bsNotesFile" type="file" accept="application/json,.json" hidden></div><div id="bsNoteStatus" class="bs-notice" aria-live="polite"></div><div id="bsNotes"></div></div></section>';
  replay.append(lower);
  const moments=review.moments || [];
  document.getElementById('bsMomentKind').insertAdjacentHTML('beforeend',[...new Set(moments.map(m=>m.kind))].map(k=>`<option>${esc(k)}</option>`).join(''));
  function renderMoments(){
    const kind=document.getElementById('bsMomentKind').value,camp=document.getElementById('bsMomentCamp').value,q=document.getElementById('bsMomentSearch').value.toLowerCase();
    const filtered=moments.filter(m=>(!kind||m.kind===kind)&&(!camp||m.camp===camp)&&`${m.title} ${m.detail} ${JSON.stringify(m.raw_records||[])}`.toLowerCase().includes(q));
    document.getElementById('bsMomentCount').textContent=`${filtered.length} 个过程 / 转折点`;
    document.getElementById('bsMoments').innerHTML=filtered.map(m=>`<article class="bs-event-group"><button type="button" class="bs-moment" data-seek="${m.time}" data-title="${esc(m.title)}"><time>${formatTime(m.time)}${m.end>m.time?'–'+formatTime(m.end):''}</time><span><strong>${esc(m.title)}</strong><small>${esc(m.detail)} · 回看前3秒上下文</small></span></button><details class="bs-raw-records"><summary>原始记录 · ${(m.raw_records||[]).length}条</summary><pre>${esc(JSON.stringify(m.raw_records||[],null,2))}</pre></details></article>`).join('')||'<div class="bs-empty">此条件下没有关键事件</div>';
  }
  ['bsMomentKind','bsMomentCamp','bsMomentSearch'].forEach(id=>document.getElementById(id).addEventListener('input',renderMoments));
  document.getElementById('bsMoments').onclick=event=>{const el=event.target.closest('[data-seek]');if(el)seek(Number(el.dataset.seek)-3,`${el.dataset.title} · 事件时刻 ${formatTime(Number(el.dataset.seek))} · 从前3秒回看`);};renderMoments();
  const noteKey=`battlescope:notes:v2:${data.match.game_id}`;let notes=[];
  const validNote=n=>n && Number.isFinite(n.time) && n.time>=0 && n.time<=Number(data.match.时长秒||maxT) && typeof n.text==='string' && n.text.length>0 && n.text.length<=2000;
  try {const stored=JSON.parse(localStorage.getItem(noteKey)||'[]');if(Array.isArray(stored))notes=stored.filter(validNote).slice(0,200);}catch{}
  function saveNotes(){try{localStorage.setItem(noteKey,JSON.stringify(notes));document.getElementById('bsNoteStatus').textContent='已保存到浏览器，可导出文件备份';}catch{document.getElementById('bsNoteStatus').textContent='浏览器存储不可用，请导出笔记备份';}renderNotes();}
  function renderNotes(){document.getElementById('bsNotes').innerHTML=notes.map((n,i)=>`<div class="bs-note"><button type="button" data-note-seek="${n.time}">${formatTime(n.time)} 跳转</button><button type="button" data-note-delete="${i}" aria-label="删除${formatTime(n.time)}的笔记">删除</button><p>${esc(n.text)}</p></div>`).join('')||'<p class="bs-muted">播放到关键时刻，记录你的判断。</p>';}
  document.getElementById('bsAddNote').onclick=()=>{const text=document.getElementById('bsNoteText').value.trim();if(!text)return;if(notes.length>=200){document.getElementById('bsNoteStatus').textContent='单局最多200条笔记，请先导出或删除部分笔记';return;}notes.push({time:Number(Number(slider.value).toFixed(2)),text});notes.sort((a,b)=>a.time-b.time);document.getElementById('bsNoteText').value='';saveNotes();};
  document.getElementById('bsNotes').onclick=event=>{const seekEl=event.target.closest('[data-note-seek]'),del=event.target.closest('[data-note-delete]');if(seekEl)seek(Number(seekEl.dataset.noteSeek));if(del){notes.splice(Number(del.dataset.noteDelete),1);saveNotes();}};
  async function download(payload,name,kind){
    if(delivery.mode==='backend' && delivery.export_url && location.protocol!=='file:'){
      try{
        const response=await fetch(delivery.export_url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,notes})});
        const result=await response.json();if(!response.ok)throw Error(result.error||'导出失败');
        exportStatus.innerHTML=`已保存：${esc(result.path)} · <a class="bs-link" href="${esc(result.url)}" target="_blank" rel="noopener">打开文件</a>`;
        document.getElementById('bsNoteStatus').textContent='导出成功，文件已保存到本地输出目录';
      }catch(error){exportStatus.textContent=`导出失败：${error.message}`;}
      return;
    }
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
    exportStatus.textContent='已请求浏览器下载。若内嵌浏览器未下载，请用 Chrome / Edge 打开此离线回放。';
  }
  document.getElementById('bsExportNotes').onclick=()=>download({game_id:data.match.game_id,notes},`复盘笔记-${data.match.game_id}.json`,'notes');
  document.getElementById('bsExport').onclick=()=>download({match:data.match,review,scores,timeseries_scores:data.timeseries_scores,notes},`战绩-${data.match.game_id}.json`,'scorecard');
  document.getElementById('bsImportNotes').onclick=()=>document.getElementById('bsNotesFile').click();
  document.getElementById('bsNotesFile').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>1000000)throw Error('笔记文件不能超过1MB');const payload=JSON.parse(await file.text());if(String(payload.game_id)!==String(data.match.game_id)||!Array.isArray(payload.notes)||payload.notes.length>200||!payload.notes.every(validNote))throw Error('笔记格式或比赛编号不匹配');for(const n of payload.notes){if(!notes.some(old=>old.time===n.time&&old.text===n.text))notes.push({time:n.time,text:n.text});}notes=notes.sort((a,b)=>a.time-b.time).slice(0,200);saveNotes();}catch(error){document.getElementById('bsNoteStatus').textContent=error.message;}event.target.value='';};renderNotes();
  document.addEventListener('keydown',event=>{if(replay.hidden||event.ctrlKey||event.altKey||event.metaKey||/INPUT|TEXTAREA|SELECT|BUTTON/.test(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();if(timer||animationFrame!==null)stop();else startSmoothPlayback(()=>Number(customSpeed.value)||1,customPlay);}else if(event.code==='ArrowLeft'){event.preventDefault();seek(Number(slider.value)-(event.shiftKey?15:5));}else if(event.code==='ArrowRight'){event.preventDefault();seek(Number(slider.value)+(event.shiftKey?15:5));}});
  replay.insertAdjacentHTML('beforeend','<div class="bs-keyboard">快捷键：空格 播放/暂停 · ← → 前后5秒 · Shift + ← → 前后15秒</div>');
  if(location.hash.includes('t='))switchView('replay');else switchView(scores.length?'overview':'replay');
})();
