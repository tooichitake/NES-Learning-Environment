import { $ } from './js/dom.js';
import { enhanceSelect } from './js/select.js';
import { blit, setMonitorMinWidth } from './js/rendering.js';
import { download } from './js/downloads.js';
import {
  applyPanelLayout,
  configurePanels,
  initControlsTabs,
  initFloatingPanels,
  initViewMenu,
  isPanelVisible,
  panelList,
  refreshControlsTabs,
  refreshPanelToggles,
  resetPanelLayout,
} from './js/panels.js';

window.resetPanelLayout = resetPanelLayout;

const BTN = { A:0x01, B:0x02, SELECT:0x04, START:0x08, UP:0x10, DOWN:0x20, LEFT:0x40, RIGHT:0x80 };
// Keyboard layouts are local: this browser's first owned controller gets layout 1, second gets layout 2.
const KEY_LAYOUTS = [
  {
    keys: { w:BTN.UP, s:BTN.DOWN, a:BTN.LEFT, d:BTN.RIGHT, g:BTN.A, f:BTN.B, '1':BTN.SELECT, '2':BTN.START },
    hints: { [BTN.UP]:'W', [BTN.DOWN]:'S', [BTN.LEFT]:'A', [BTN.RIGHT]:'D', [BTN.B]:'F', [BTN.A]:'G', [BTN.SELECT]:'1', [BTN.START]:'2' },
  },
  {
    keys: { ArrowUp:BTN.UP, ArrowDown:BTN.DOWN, ArrowLeft:BTN.LEFT, ArrowRight:BTN.RIGHT, ',':BTN.B, '.':BTN.A, '/':BTN.SELECT, Enter:BTN.START },
    hints: { [BTN.UP]:'↑', [BTN.DOWN]:'↓', [BTN.LEFT]:'←', [BTN.RIGHT]:'→', [BTN.B]:',', [BTN.A]:'.', [BTN.SELECT]:'/', [BTN.START]:'↵' },
  },
];
// Debug shortcuts: one keyboard row per port, action index i -> row[i]; '0' maps to the last action.
const RL_ACTION_KEYS = [
  ['1','2','3','4','5','6','7','8','9','0'],
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],
  ['z','x','c','v','b','n','m',',','.','/'],
];
const statusEl = $('status');
function setStatus(t, k){ if(!statusEl) return; statusEl.textContent = t; statusEl.hidden = !t; statusEl.className = 'status glass ' + (k||''); }
const native=$('native'), nctx=native.getContext('2d');
const obs=$('obs'), octx=obs.getContext('2d');

// ---- Mode ----
// Play is real time. Debug is step driven. Agent is real time with agent peers.
let mode='play';
const modeSeg=$('modeSeg'), segInd=$('segInd');
configurePanels({ getMode: () => mode, refreshVis: () => refreshVis() });
function positionSegInd(){ const b=modeSeg.querySelector('button.active'); if(!b) return; segInd.style.left=b.offsetLeft+'px'; segInd.style.width=b.offsetWidth+'px'; }
function refreshVis(){
  document.body.dataset.mode = mode;
  document.querySelectorAll('[data-modes]').forEach(el=>{
    const show = el.dataset.modes.split(' ').includes(mode);
    const panelId=el.dataset.panel;
    const panelOk=!panelId || isPanelVisible(panelId);
    el.hidden = !(show && panelOk);
  });
  for(const panel of panelList()){
    if(!panel.dataset.modes) panel.hidden = !isPanelVisible(panel.dataset.panel);
  }
  refreshControlsTabs();
  refreshPanelToggles();
  requestAnimationFrame(()=>{ positionSegInd(); applyPanelLayout(); requestAnimationFrame(applyPanelLayout); });
}
// Controllers (NES hardware) exist in Play + Agent; Debug is action-space stepping (no ports).
function controllersActive(){ return romLoaded && (mode==='play' || mode==='human'); }
// owners[p] is server-authoritative: {client_id, role, name}.
function ownerClientId(owner){ return owner ? owner.client_id : null; }
function ownsMe(p){ return ownerClientId(owners[p])===myId; }
function drivesPort(p){ return controllersActive() && ownsMe(p); }
function localPorts(){
  if(!controllersActive()) return [];
  const ports=[];
  for(let p=0;p<UI_PORTS;p++) if(ownsMe(p)) ports.push(p);
  return ports;
}
function keyboardLayoutForPort(port, ports=localPorts()){
  const slot=ports.indexOf(port);
  return slot>=0 && slot<KEY_LAYOUTS.length ? KEY_LAYOUTS[slot] : null;
}
// Show pads only for ports driven by this browser.
function refreshPads(){
  let any=false;
  const ports=localPorts();
  for(let p=0;p<UI_PORTS;p++){
    const own=drivesPort(p);
    if(own) any=true;
    else { pressed[p].clear(); held[p]=0; }
    const w=$('padwrap'+p); if(w) w.style.display = own ? '' : 'none';
    updatePadKeyHints(p, keyboardLayoutForPort(p, ports));
  }
  syncPads();
  const g=$('padGroup'); if(g && (mode==='play'||mode==='human')) g.style.display = any ? '' : 'none';
}
function applyModeSettings(){
  setAuto(false);
  if(mode==='play'){ send({t:'settings', rl_mode:false, step_mode:false}); }
  else if(mode==='rl'){ send({t:'settings', rl_mode:true, step_mode:true,
    frame_skip:+$('frameSkip').value, obs_size:+$('obsSize').value, maxpool:$('maxpoolTgl').checked,
    remove_sprite_limit:$('spriteTgl').checked, obs_rgb:$('rgbTgl').checked,
    terminal_on_life_loss:$('lifeLoss').checked,
    clip_pos:+$('clipPos').value, clip_neg:+$('clipNeg').value,
    sticky_prob:+$('stickyProb').value, noop_max:+$('noopMax').value }); }
  else { send({t:'settings', rl_mode:false, step_mode:false}); }
}
function setMode(m){
  mode=m;
  [...modeSeg.querySelectorAll('button')].forEach(b=>{
    const active=b.dataset.mode===m;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', String(active));
    b.tabIndex = active ? 0 : -1;
  });
  positionSegInd();
  $('obsCap').textContent = 'Frame';
  refreshVis();
  applyModeSettings();
  if(romLoaded){ applyControllers(); refreshCtl(); }
  if(m==='human') renderAgent();
  for(let p=0;p<UI_PORTS;p++) pressed[p].clear(); held=[0,0,0,0]; refreshPads(); syncPads();
}
function applyControllers(){
  if(!romLoaded) return;
  if(mode==='play'){
    let mine=false; for(let p=0;p<UI_PORTS;p++) if(ownsMe(p)) mine=true;
    if(!mine){ for(let p=0;p<visiblePorts();p++) if(owners[p]==null){ assignPort(p, myId, suggestPlayerName(peerById(myId), p)); break; } }
  } else if(mode==='rl'){
    for(let p=0;p<UI_PORTS;p++) if(ownsMe(p)) assignPort(p, null);
  }
}

// ---- WebSocket thin client ----
let ws=null, romLoaded=false, running=true;
let myId=null, players=1, owners=[null,null,null,null], peers=[], lastPortsKey='', myLabel='';
let actx=null, nextAudioTime=0, recording=false, romName='', currentRomBytes=null, uploadedGameId=null;
let stepMasks=[]; // Debug: one selected action mask per controller port (multi-agent)
let actionMasks=[], actionNames=[];
let gamesRoster=[]; // welcome roster [{id, display_name, players, sha1, levels}] for ROM auto-detect
let focusedAgentId=null, liveAgents=[];
// NES Four Score cap. Per-controller input: pressed[port] = held button bits, held[port] = last mask sent.
const UI_PORTS=4;
const pressed=[new Set(), new Set(), new Set(), new Set()];
let held=[0,0,0,0];
// Always surface all 4 ports as claimable (the hardware cap, not the game's player count).
function visiblePorts(){ return UI_PORTS; }

function levelsForGame(game){
  const g=gamesRoster.find(x=>x.id===game);
  return (g && Array.isArray(g.levels)) ? g.levels.slice() : [];
}
function buildLevels(levels, selected){
  const sel=$('levelSel'); if(!sel) return;
  const prior=selected || sel.value || '';
  const list=(levels&&levels.length?levels:[]).slice();
  sel.innerHTML='';
  list.forEach(level=>{ const o=document.createElement('option'); o.value=level.id; o.textContent=level.label; sel.appendChild(o); });
  if(!list.length){ const o=document.createElement('option'); o.value=''; o.textContent='Unavailable'; sel.appendChild(o); }
  // Default to the earliest real LEVEL, not "title"; we test from a level start state.
  const firstReal=list.find(level=>level.id!=='title');
  sel.value = list.some(level=>level.id===prior) ? prior : (firstReal?.id || list[0]?.id || '');
  enhanceSelect(sel); if(sel._sync) sel._sync();
}
// Game -> Mode -> Level menu: specs sharing a `display_name` are one game's modes.
let gameGroups=new Map(); // display_name -> [specs]
function rebuildGroups(){
  gameGroups=new Map();
  gamesRoster.forEach(g=>{ if(!gameGroups.has(g.display_name)) gameGroups.set(g.display_name, []); gameGroups.get(g.display_name).push(g); });
}
function selectedSpec(){
  // Key on gym_id (unique per game+mode), not the shared `id`, else 1P/2P both match.
  const gid=$('modeSel') ? $('modeSel').value : '';
  return gamesRoster.find(g=>g.gym_id===gid) || null;
}
function buildModes(displayName, selectedId){
  const sel=$('modeSel'); if(!sel) return;
  const group=gameGroups.get(displayName)||[];
  sel.innerHTML='';
  // Mode is always resident: a single-mode game shows one `None` option, multi-mode games list each.
  group.forEach(g=>{ const o=document.createElement('option'); o.value=g.gym_id; o.textContent=g.mode||'None'; sel.appendChild(o); });
  sel.value = group.some(g=>g.gym_id===selectedId) ? selectedId : (group[0]?.gym_id || '');
  enhanceSelect(sel); if(sel._sync) sel._sync();
}
function refreshLevels(selected){
  const spec=selectedSpec();
  buildLevels(spec ? levelsForGame(spec.id) : [], selected);
}
// Auto-load the env: the server resolves the packaged ROM by sha1; an uploaded ROM is reused only within the same game.
function reloadSelectedEnv(){
  const spec=selectedSpec(); const envId=envIdForGame(spec);
  if(!spec || !envId) return;
  // Kill a carried-over auto-step timer so a new game/mode/level never silently auto-plays.
  setAuto(false);
  setStatus('Level: '+($('levelSel').selectedOptions[0]?.textContent || selectedLevel()), 'ok');
  if(currentRomBytes && uploadedGameId===spec.id){
    send({ t:'load_rom', env_id:envId, bytes_b64: bytesToBase64(currentRomBytes) });
  } else {
    currentRomBytes=null; uploadedGameId=null;
    send({ t:'load_rom', env_id:envId });
  }
}
function selectedLevel(){
  return ($('levelSel') && $('levelSel').value) || '';
}
function envIdForGame(game){
  const level=selectedLevel();
  if(!game || !level) return '';
  if(level==='title') return game.players===1 ? game.gym_id+'-v0' : game.gym_id;
  if(game.players===1) return game.gym_id+'-'+level+'-v0';
  return game.gym_id.replace(/-v0$/, '')+'-'+level+'-v0';
}
function levelFromEnvId(game, envId){
  if(!game || !envId) return '';
  if(envId === (game.players===1 ? game.gym_id+'-v0' : game.gym_id)) return 'title';
  return levelsForGame(game.id).find(level=>{
    if(level.id==='title') return false;
    const stem = game.players===1 ? game.gym_id+'-'+level.id : game.gym_id.replace(/-v0$/, '')+'-'+level.id;
    // Require the -v/NoFrameskip suffix so level "1" doesn't prefix-match "10" / "1-1".
    return envId.startsWith(stem+'-v') || envId.startsWith(stem+'NoFrameskip');
  })?.id || '';
}
function bytesToBase64(buf){
  let bin='';
  for(let i=0;i<buf.length;i+=0x8000) bin+=String.fromCharCode.apply(null, buf.subarray(i,i+0x8000));
  return btoa(bin);
}

function connect(){
  const url=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws';
  ws=new WebSocket(url); ws.binaryType='arraybuffer';
  ws.onopen=()=>{
    send({t:'hello', role:'human'});
    setStatus('', '');
  };
  ws.onclose=()=>{ setStatus('Disconnected - retrying...','err'); setTimeout(connect,1000); };
  ws.onerror=()=>setStatus('Connection error','err');
  ws.onmessage=onMessage;
}
function send(o){ if(ws&&ws.readyState===1) ws.send(JSON.stringify(o)); }

function buildGames(games){
  if(games&&games.length) gamesRoster=games.slice(); // keep for ROM SHA-1 auto-detect
  rebuildGroups();
  const sel=$('gameSel'); if(!games||!games.length||sel.dataset.built) return; sel.dataset.built='1';
  sel.innerHTML='';
  // Alphabetical (English) game order; one entry per display_name (modes live in modeSel).
  const names=[...new Set(gamesRoster.map(g=>g.display_name))].sort((a,b)=>a.localeCompare(b));
  names.forEach(name=>{ const o=document.createElement('option'); o.value=name; o.textContent=name; sel.appendChild(o); });
  enhanceSelect(sel); if(sel._sync) sel._sync();
  buildModes(sel.value);
  refreshLevels();
  // Auto-load the first game's level on startup so the live frame shows immediately.
  reloadSelectedEnv();
}

function onMessage(ev){
  if(typeof ev.data==='string'){ const m=JSON.parse(ev.data);
    if(m.t==='welcome'){ myId=m.client_id; buildGames(m.games); }
    else if(m.t==='ready') onReady(m);
    else if(m.t==='ports') updatePorts(m.players, m.owners, m.peers||[]);
    else if(m.t==='error'){
      // Auto-load found no packaged ROM -> fall back to Upload ROM (menu stays on the requested game).
      if(m.code==='rom_required') setStatus(m.message || 'ROM not packaged - click Upload ROM', 'wait');
      else setStatus(m.message || m.code || 'Server error', 'err');
    }
    else if(m.t==='recording') saveRecording(m);
    else if(m.t==='ram') saveRam(m);
    return; }
  const u8=new Uint8Array(ev.data), dv=new DataView(ev.data);
  const mlen=dv.getUint32(0,true);
  const meta=JSON.parse(new TextDecoder().decode(u8.subarray(4,4+mlen)));
  let off=4+mlen;
  const rgbN=meta.native_w*meta.native_h*3; const rgb=u8.subarray(off,off+rgbN); off+=rgbN;
  const ch=meta.obs_channels||1; const obsN=meta.obs_w*meta.obs_h*ch; const og=u8.subarray(off,off+obsN); off+=obsN;
  // Debug is muted (obs is video-only); Play + Agent play sound.
  if(meta.audio_len>0 && meta.audio_rate>0 && mode!=='rl') playAudio(ev.data, off, meta.audio_len, meta.audio_rate);
  blit(native,nctx,rgb,meta.native_w,meta.native_h,3);
  if(mode==='human'){
    // Agent: one card per agent, each blitting its own obs block (meta.agents order) after the audio.
    renderAgentGrid(meta, u8, off + (meta.audio_len||0)*4);
  } else if(meta.obs_step !== false){
    // Debug: the single stepped-agent obs + per-port stats below (monitor reports outcomes only).
    blit(obs,octx,og,meta.obs_w,meta.obs_h,ch);
    const live=$('iLive'); if(live){ live.textContent='Live'; live.className='status-pill live'; }
    const done=$('iDone');
    if(done){
      const ended=!!meta.terminated || !!meta.truncated;
      done.textContent=ended ? (meta.truncated?'Truncated':'Terminal') : 'Running';
      done.className='status-pill '+(ended?'bad':'');
    }
    $('iEp').textContent=meta.step||0;
    renderAgentStats(meta);
  }
  if(meta.recording!==undefined) setRecUI(meta.recording);
}
// Agent: each agent gets a debug-style card backed by its own env profile + obs block.
const agentCards = new Map(); // client id -> {root, canvas, ctx, label, port, rew, ret, lives, env, act}
function envSummary(env){
  if(!env) return 'env not declared';
  return env.env_id || 'env not declared';
}
function agentActionName(agent, mask){
  const masks=agent.action_masks || [];
  const names=agent.action_names || [];
  const i=masks.indexOf(mask);
  if(i>=0) return (i+':'+(names[i]||''));
  return mask ? ('0x'+mask.toString(16)) : 'NOOP';
}
// Per-agent reward/return/lives breakout for multi-player envs (shared screen above).
let lastAgentRewards=[];
function renderAgentStats(meta){
  const as=$('agentStats'); if(!as) return;
  const n=Math.max(1, players);
  let rows='<div class="agent-stat agent-stat-head"><span>Agent</span><span>Reward</span><span>Return</span><span>Lives</span></div>';
  const next=[];
  for(let p=0;p<n;p++){
    const r=(meta.rewards||[])[p]||0, rt=(meta.rets||[])[p]||0, lv=(meta.lives||[])[p]||0;
    next[p]=r;
    const changed = lastAgentRewards[p]!==undefined && r!==lastAgentRewards[p] && r!==0;
    const pulse = changed ? (r>0?' pulse-pos':' pulse-neg') : '';
    rows+='<div class="agent-stat'+pulse+'"><span class="as-name">P'+(p+1)+'</span>'
      +'<span class="reward '+(r>0?'pos':r<0?'neg':'')+'">'+(r>=0?'+':'')+r.toFixed(2)+'</span>'
      +'<span>'+rt.toFixed(1)+'</span><span>'+lv+'</span></div>';
  }
  as.innerHTML=rows; as.style.display=''; lastAgentRewards=next;
}
function renderActionChips(container, masks, names, activeMask){
  if(!container) return;
  const key=JSON.stringify([masks || [], names || []]);
  if(container.dataset.key!==key){
    container.dataset.key=key;
    container.innerHTML='';
    (names || []).forEach((name,i)=>{
      const c=document.createElement('span');
      c.className='chip';
      c.dataset.mask=(masks || [])[i] || 0;
      c.textContent=i+':'+name;
      container.appendChild(c);
    });
  }
  for(const c of container.children) c.classList.toggle('active', (+c.dataset.mask|0)===(activeMask|0));
}
function makeAgentCard(){
  const root=document.createElement('div'); root.className='agentcard agentdebug';
  root.tabIndex=0;
  root.setAttribute('role','button');
  root.setAttribute('aria-pressed','false');
  root.innerHTML='<div class="monitor-strip"><span class="status-pill live alabel"></span><span class="status-pill aport"></span></div>'
    +'<h3 class="aenv">env not declared</h3>'
    +'<div class="obswrap"><canvas class="acanvas" width="84" height="84"></canvas></div>'
    +'<div class="stats">'
    +'<div class="tile"><strong>Step</strong><span class="aep">0</span></div>'
    +'<div class="tile"><strong>Reward</strong><span class="arew reward">0</span></div>'
    +'<div class="tile"><strong>Return</strong><span class="aret">0</span></div>'
    +'<div class="tile"><strong>Lives</strong><span class="alives">0</span></div>'
    +'</div>'
    +'<div class="action-monitor"><div class="action-line"><span>Current Action</span><strong class="aact">NOOP</strong></div><div class="chips achips"></div></div>';
  const canvas=root.querySelector('.acanvas');
  root.addEventListener('click', ()=>setFocusedAgent(root.dataset.agentId || null));
  root.addEventListener('keydown', e=>{
    if(e.key==='Enter' || e.key===' '){ setFocusedAgent(root.dataset.agentId || null); e.preventDefault(); }
  });
  return { root, canvas, ctx: canvas.getContext('2d'),
    label: root.querySelector('.alabel'), port: root.querySelector('.aport'),
    ep: root.querySelector('.aep'), rew: root.querySelector('.arew'), ret: root.querySelector('.aret'),
    lives: root.querySelector('.alives'), env: root.querySelector('.aenv'), act: root.querySelector('.aact'),
    chips: root.querySelector('.achips') };
}
function renderAgentStatus(n){
  const bar=$('agentBar'); if(!bar) return;
  bar.classList.toggle('connected', n>0);
  $('agentText').textContent = n>0 ? (n+' agent'+(n>1?'s':'')+' connected') : 'Waiting for an agent...';
}
function setFocusedAgent(id){
  focusedAgentId = focusedAgentId===id ? null : id;
  syncAgentFocus();
}
function syncAgentFocus(){
  const grid=$('agentGrid'); if(!grid) return;
  if(focusedAgentId && !agentCards.has(focusedAgentId)) focusedAgentId=null;
  grid.classList.toggle('has-focus', !!focusedAgentId);
  for(const [id, card] of agentCards){
    const focused=id===focusedAgentId;
    card.root.classList.toggle('focused', focused);
    card.root.setAttribute('aria-pressed', String(focused));
  }
}
function updateAgentGridSizing(agents){
  const grid=$('agentGrid'); if(!grid) return;
  let maxW=84;
  for(const a of agents) maxW=Math.max(maxW, a.obs_w|0);
  const cardMin=Math.max(300, Math.min(420, maxW + 72));
  grid.style.setProperty('--agent-card-min', cardMin+'px');
  setMonitorMinWidth(cardMin + 36);
}
function clearAgentGrid(){
  const g=$('agentGrid');
  if(g){ g.innerHTML=''; g.style.removeProperty('--agent-card-min'); }
  setMonitorMinWidth(260);
  focusedAgentId=null;
  liveAgents=[];
  agentCards.clear();
}
function renderAgentGrid(meta, u8, off){
  const grid=$('agentGrid'); if(!grid) return;
  const agents = meta.agents || [];
  liveAgents = agents.slice();
  renderAgentStatus(agents.length);
  updateAgentGridSizing(agents);
  // Drop cards for agents that left.
  const present = new Set(agents.map(a=>a.id));
  for(const id of [...agentCards.keys()]) if(!present.has(id)){ agentCards.get(id).root.remove(); agentCards.delete(id); if(focusedAgentId===id) focusedAgentId=null; }
  for(const a of agents){
    const n = a.obs_w*a.obs_h*a.obs_channels; const block = u8.subarray(off, off+n); off += n;
    let card = agentCards.get(a.id);
    if(!card){ card = makeAgentCard(); agentCards.set(a.id, card); grid.appendChild(card.root); }
    card.root.dataset.agentId=String(a.id);
    card.root.setAttribute('aria-label', (a.label || ('Agent '+a.id))+' monitor');
    card.label.textContent = a.label || ('Agent '+a.id);
    card.port.textContent = a.assigned ? ('P'+(((a.port|0))+1)) : 'Unassigned';
    card.port.classList.toggle('warn', !a.assigned);
    card.env.textContent = envSummary(a.env);
    card.act.textContent = agentActionName(a, a.mask|0);
    card.act.className = 'aact '+((a.mask|0)?'':'idle');
    renderActionChips(card.chips, a.action_masks || [], a.action_names || [], a.mask|0);
    if(a.obs_step){ // refresh the obs + reward only on this agent's own window boundary
      blit(card.canvas, card.ctx, block, a.obs_w, a.obs_h, a.obs_channels);
      const r=+a.reward||0; card.rew.textContent=(r>=0?'+':'')+r.toFixed(2); card.rew.className='arew '+(r>0?'pos':r<0?'neg':'');
      card.ep.textContent=String(a.step||0);
      card.ret.textContent=(+a.ret||0).toFixed(1);
      card.lives.textContent=String(a.lives|0);
    }
  }
  syncAgentFocus();
}
// Agent presence before frames arrive: derive from port owners; live frames refine the count.
function renderAgent(){
  const n = Math.max(liveAgents.length, peers.filter(p=>p.role==='agent').length);
  renderAgentStatus(n);
}
function onReady(m){
  // Match on id AND player count: 1P + multi-P specs share `id`, so the count disambiguates.
  const readyGame=gamesRoster.find(g=>g.id===m.game && g.players===(m.players||g.players))
    || gamesRoster.find(g=>g.id===m.game);
  if(readyGame && $('gameSel')){
    if($('gameSel').value!==readyGame.display_name){ $('gameSel').value=readyGame.display_name; if($('gameSel')._sync) $('gameSel')._sync(); }
    buildModes(readyGame.display_name, readyGame.gym_id);
  }
  refreshLevels(levelFromEnvId(readyGame, m.env_id));
  romLoaded=true; running=true; players=m.players||1;
  // End-on-life-loss default mirrors the env: on for single-agent (ALE), off for multi-player last-standing.
  if($('lifeLoss')) $('lifeLoss').checked = (players===1);
  setStatus('', '');
  ['btnRun','btnReset','btnReset2','btnRec','btnRam'].forEach(id=>{ if($(id)) $(id).disabled=false; });
  $('btnRun').querySelector('span').textContent='Pause'; $('btnRun').className='btn btn-good';
  actionMasks=(m.action_masks||[]).slice(); actionNames=(m.actions||[]).slice();
  // Debug: per-port action selectors + a Step button (multi-agent steps all ports together).
  buildActionInputs();
  clearAgentGrid();
  refreshCtl(); refreshPads();
  setMode(mode); // applies the mode's controller ownership (claims/releases) + visibility
}
// Debug input: one action-selector row per port + a Step button (one action per port, then ONE step).
function buildActionInputs(){
  const ab=$('actBtns'); if(!ab) return;
  ab.innerHTML='';
  const n=Math.max(1, players);
  stepMasks=new Array(n).fill(0);
  for(let p=0;p<n;p++){
    const row=document.createElement('div'); row.className='agent-actrow';
    if(n>1){ const lbl=document.createElement('span'); lbl.className='agent-actlabel'; lbl.textContent='P'+(p+1); row.appendChild(lbl); }
    const btns=document.createElement('div'); btns.className='btnrow agent-actbtns'; btns.dataset.port=p;
    const keys=RL_ACTION_KEYS[p]||[];
    actionNames.forEach((name,i)=>{
      const b=document.createElement('button'); b.className='actbtn'+(i===0?' hot':''); b.dataset.mask=actionMasks[i]||0;
      const lbl=document.createElement('span'); lbl.textContent=i+':'+name; b.appendChild(lbl);
      if(keys[i]){ const k=document.createElement('span'); k.className='actkey'; k.textContent=keys[i].toUpperCase(); b.appendChild(k); }
      b.addEventListener('click',()=>selectAction(p, actionMasks[i]||0, btns)); btns.appendChild(b);
    });
    row.appendChild(btns); ab.appendChild(row);
  }
  const step=document.createElement('button'); step.id='btnStepAll'; step.type='button'; step.className='btn btn-primary step-all';
  const slbl=document.createElement('span'); slbl.textContent=n>1?'Step (all agents)':'Step'; step.appendChild(slbl);
  const sk=document.createElement('span'); sk.className='actkey'; sk.textContent='Space'; step.appendChild(sk);
  step.addEventListener('click',sendStep); ab.appendChild(step);
}
function selectAction(port, mask, btnsEl){
  stepMasks[port]=mask;
  for(const b of btnsEl.children) b.classList.toggle('hot', (+b.dataset.mask|0)===(mask|0));
  if(Math.max(1,players)<=1) sendStep(); // single-agent: a click is select + step
}
function sendStep(){
  resumeAudio();
  send({t:'step', masks: stepMasks.slice(0, Math.max(1,players)).map(int8)});
}
function int8(x){ return x & 0xff; }

function updatePorts(np, no, peerList){
  // `no` entries are controller-player instances, not raw websocket peers.
  const peerKey=peerList.map(p=>p.id+':'+p.role+':'+p.name+':'+(p.env_id||'')).join(',');
  const key=np+'|'+no.map(o=>o?[o.client_id,o.role,o.name].join(':'):'-').join(',')+'|'+peerKey;
  if(key===lastPortsKey) return; lastPortsKey=key;
  players=np; owners=no.slice(); peers=peerList.slice();
  const me=peerById(myId);
  if(me){ myLabel=me.name||''; syncNameField(); }
  refreshCtl();
  refreshPads();
  if(mode==='human') renderAgent(); // ownership changes can flip the agent presence state
}
function peerById(id){ return peers.find(p=>p.id===id) || null; }
function roleLabel(peer){ if(!peer) return 'Free'; return ownerClientId(peer)===myId ? 'You' : (peer.role==='agent' ? 'Agent' : 'Human'); }
function roleClass(peer){ if(!peer) return 'free'; return ownerClientId(peer)===myId ? 'you' : peer.role; }
function assignPort(port, clientId, name){
  const msg={t:'assign_port', port, client_id:clientId==null ? null : clientId};
  if(clientId!=null) msg.name=name;
  send(msg);
}
function uniquePlayerName(base, port){
  const root=(base || 'Player').trim() || 'Player';
  const used=new Set(owners.map((owner, i)=>i===port || !owner ? null : owner.name.toLowerCase()).filter(Boolean));
  if(!used.has(root.toLowerCase())) return root;
  for(let i=2;i<100;i++){
    const name=root+' '+i;
    if(!used.has(name.toLowerCase())) return name;
  }
  return root+' '+Date.now().toString(36);
}
function suggestPlayerName(peer, port){ return uniquePlayerName(peer?.name || (peer?.role==='agent' ? 'Agent' : 'Player'), port); }
function syncNameField(){
  // Own-name fields live inline in any controller row currently assigned to this browser.
  for(let p=0;p<UI_PORTS;p++){
    const input=$('ctlName'+p);
    if(input && ownsMe(p) && document.activeElement!==input) input.value=owners[p]?.name || myLabel || 'Player';
  }
}
function renamePlayer(port, value){
  const owner=owners[port];
  if(!owner || !ownsMe(port)) return;
  const name=uniquePlayerName(value, port);
  const input=$('ctlName'+port);
  if(input) input.value=name;
  if(!name || name===owner.name) return;
  assignPort(port, ownerClientId(owner), name);
}

// Each controller is a source pop-up; the server owns the assignment table, this UI edits its own name.
function refreshCtl(){
  // All four ports (the Four Score cap) are always shown + claimable, regardless of player count.
  const vis=visiblePorts();
  for(let p=0;p<UI_PORTS;p++){
    const row=$('ctlrow'+p); if(!row) continue;
    if(p>=vis){ row.style.display='none'; continue; }
    row.style.display='';
    const owner=owners[p];
    const ownerPeer=owner ? (peerById(ownerClientId(owner)) || owner) : null;
    const role=$('ctlRole'+p), name=$('ctlName'+p), sel=$('ctlSource'+p);
    if(role){ role.textContent=roleLabel(owner || ownerPeer); role.className='role-badge '+roleClass(owner || ownerPeer); }
    if(name){
      name.value=owner ? owner.name : 'Free';
      name.readOnly=!ownsMe(p);
      name.tabIndex=ownsMe(p) ? 0 : -1;
      name.classList.toggle('editable', ownsMe(p));
    }
    if(sel){
      const value=owner ? String(ownerClientId(owner)) : '';
      sel.innerHTML='';
      const free=document.createElement('option');
      free.value=''; free.textContent='Free'; sel.appendChild(free);
      if(owner){
        const current=document.createElement('option');
        current.value=value;
        current.textContent=roleLabel(owner)+' - '+owner.name;
        sel.appendChild(current);
      }
      peers.forEach(peer=>{
        if(owner && peer.id===ownerClientId(owner)) return;
        const option=document.createElement('option');
        option.value=String(peer.id);
        const role=peer.id===myId ? 'You' : (peer.role==='agent' ? 'Agent' : 'Human');
        const name=peer.name||('Peer '+peer.id);
        option.textContent=role+' - '+name;
        sel.appendChild(option);
      });
      sel.value=value;
      enhanceSelect(sel);
      if(sel._sync) sel._sync();
    }
  }
}

$('btnUploadRom').addEventListener('click', ()=>$('romFile').click());
$('romFile').addEventListener('change', async e=>{
  const f=e.target.files[0]; if(!f) return;
  $('fileName').textContent=f.name; romName=f.name;
  setStatus('Loading ROM...','wait');
  const ab=await f.arrayBuffer(); const buf=new Uint8Array(ab); currentRomBytes=buf;
  // Prefer registry SHA-1 over the dropdown so game/player-count match the ROM.
  let selected=selectedSpec();
  try{
    const hash=await crypto.subtle.digest('SHA-1', ab);
    const hex=[...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join('');
    const m=(selected && selected.sha1 && selected.sha1.toLowerCase()===hex)
      ? selected
      : gamesRoster.find(g=>g.sha1 && g.sha1.toLowerCase()===hex);
    if(m){
      selected=m;
      // Point the Game/Mode menu at the detected spec.
      if($('gameSel').value!==m.display_name){ $('gameSel').value=m.display_name; const s=$('gameSel'); if(s._sync) s._sync(); }
      buildModes(m.display_name, m.gym_id); refreshLevels();
      setStatus('Detected '+m.display_name+(m.mode?' ('+m.mode+')':'')+' - loading...','wait');
    } else {
      const sel=$('gameSel').selectedOptions[0];
      setStatus('Unrecognized ROM - loading as '+(sel?sel.textContent:$('gameSel').value)+'...','wait');
    }
  }catch(err){ /* SubtleCrypto unavailable -> use the dropdown selection */ }
  const envId=envIdForGame(selected);
  if(selected && envId){ uploadedGameId=selected.id; send({ t:'load_rom', env_id:envId, bytes_b64: bytesToBase64(buf) }); }
});

// ---- Controls ----
[...modeSeg.querySelectorAll('button')].forEach(b=>b.addEventListener('click', ()=>{ resumeAudio(); setMode(b.dataset.mode); }));
modeSeg.addEventListener('keydown', e=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
  const buttons=[...modeSeg.querySelectorAll('button')];
  const current=Math.max(0, buttons.findIndex(b=>b.dataset.mode===mode));
  let next=current;
  if(e.key==='ArrowLeft') next=(current+buttons.length-1)%buttons.length;
  else if(e.key==='ArrowRight') next=(current+1)%buttons.length;
  else if(e.key==='Home') next=0;
  else if(e.key==='End') next=buttons.length-1;
  resumeAudio();
  setMode(buttons[next].dataset.mode);
  buttons[next].focus();
  e.preventDefault();
});
$('gameSel').addEventListener('change', ()=>{ buildModes($('gameSel').value); refreshLevels(); reloadSelectedEnv(); });
$('modeSel').addEventListener('change', ()=>{ refreshLevels(); reloadSelectedEnv(); });
$('levelSel').addEventListener('change', ()=>reloadSelectedEnv());
$('btnRun').addEventListener('click', ()=>{ resumeAudio(); running=!running; send({t:'settings', running}); $('btnRun').querySelector('span').textContent=running?'Pause':'Run'; });
['btnReset','btnReset2'].forEach(id=>$(id).addEventListener('click', ()=>send({t:'settings', reset:true})));
$('obsSize').addEventListener('input', e=>{ $('obsSizeV').textContent=e.target.value; send({t:'settings', obs_size:+e.target.value}); });
$('frameSkip').addEventListener('input', e=>{ $('frameSkipV').textContent=e.target.value; send({t:'settings', frame_skip:+e.target.value}); });
$('maxpoolTgl').addEventListener('change', e=>send({t:'settings', maxpool:e.target.checked}));
$('spriteTgl').addEventListener('change', e=>send({t:'settings', remove_sprite_limit:e.target.checked}));
$('rgbTgl').addEventListener('change', e=>send({t:'settings', obs_rgb:e.target.checked}));
// Numeric behavior knobs relabel live and take effect on the next step.
$('lifeLoss').addEventListener('change', e=>send({t:'settings', terminal_on_life_loss:e.target.checked}));
$('clipPos').addEventListener('input', e=>{ const v=+e.target.value; $('clipPosV').textContent=v?('+'+v):'off'; send({t:'settings', clip_pos:v}); });
$('clipNeg').addEventListener('input', e=>{ const v=+e.target.value; $('clipNegV').textContent=v?('-'+v):'off'; send({t:'settings', clip_neg:v}); });
$('stickyProb').addEventListener('input', e=>{ const v=+e.target.value; $('stickyProbV').textContent=v.toFixed(2); send({t:'settings', sticky_prob:v}); });
$('noopMax').addEventListener('input', e=>{ const v=+e.target.value; $('noopMaxV').textContent=String(v); send({t:'settings', noop_max:v}); });
$('btnRec').addEventListener('click', ()=>{ resumeAudio(); send({t:'record', on:!recording}); });
$('btnRam').addEventListener('click', ()=>send({t:'dump_ram'}));
function setRecUI(on){ recording=on; const b=$('btnRec'); b.querySelector('span').textContent=on?'Stop & Download':'Record'; b.className='btn'+(on?' btn-danger':''); }

for(let p=0;p<UI_PORTS;p++){
  const sel=$('ctlSource'+p);
  if(sel) sel.addEventListener('change', e=>{
    const peer=e.target.value ? peerById(Number(e.target.value)) : null;
    assignPort(p, peer ? peer.id : null, peer ? suggestPlayerName(peer, p) : null);
  });
  const input=$('ctlName'+p);
  if(input){
    input.addEventListener('change', e=>renamePlayer(p, e.target.value));
    input.addEventListener('keydown', e=>{
      if(e.key==='Enter'){ e.currentTarget.blur(); }
      else if(e.key==='Escape'){ syncNameField(); refreshCtl(); e.currentTarget.blur(); }
    });
  }
}

// Auto-step sends random Debug actions at the chosen step rate.
let autoTimer=null;
function setAuto(on){
  clearInterval(autoTimer); autoTimer=null; if($('autoTgl')) $('autoTgl').checked=on;
  if(on && mode==='rl'){ const rate=+$('stepRate').value;
    autoTimer=setInterval(()=>{ if(!actionMasks.length) return; const n=Math.max(1,players); for(let p=0;p<n;p++) stepMasks[p]=actionMasks[Math.floor(Math.random()*actionMasks.length)]; sendStep(); }, Math.max(33,1000/rate)); }
}
$('autoTgl').addEventListener('change', e=>setAuto(e.target.checked));
$('stepRate').addEventListener('input', e=>{ $('stepRateV').textContent=e.target.value; if($('autoTgl').checked) setAuto(true); });

// ---- Web Audio ----
// Jitter buffer (s) kept ahead of the playback cursor so WS timing jitter doesn't underrun (crackle).
const AUDIO_LATENCY = 0.07;
const AUDIO_RATE = 44100; // the console's APU rate; matching the context avoids per-buffer resample clicks
function resumeAudio(){
  if(!actx){
    // Create the context AT the APU's sample rate so each frame buffer plays natively (no resample crackle).
    const AC = window.AudioContext||window.webkitAudioContext; if(!AC) return;
    try{ actx=new AC({sampleRate:AUDIO_RATE, latencyHint:'interactive'}); }
    catch(e){ try{ actx=new AC(); }catch(e2){ return; } }
  }
  if(actx.state==='suspended') actx.resume();
}
function playAudio(buf, byteOff, n, rate){
  if(!actx || actx.state!=='running') return;
  const f32=new Float32Array(buf.slice(byteOff, byteOff+n*4));
  const ab=actx.createBuffer(1, n, rate); ab.getChannelData(0).set(f32);
  const src=actx.createBufferSource(); src.buffer=ab; src.connect(actx.destination);
  const now=actx.currentTime;
  // Resync on an underrun or a large drift ahead (background throttling); else append back-to-back.
  if(nextAudioTime < now + 0.005 || nextAudioTime > now + 0.5) nextAudioTime = now + AUDIO_LATENCY;
  src.start(nextAudioTime); nextAudioTime += ab.duration;
}

function saveRecording(m){
  const doc={ game_id:m.game, rom:romName||((m.game||'game')+'.nes'), frameskip:m.frameskip||1, actions:m.actions||[] };
  download('replay-'+(m.game||'game')+'-'+(m.frames||0)+'f.json', new Blob([JSON.stringify(doc)],{type:'application/json'}));
  setStatus('Recorded '+(m.frames||0)+' frames; downloaded replay JSON.','ok');
}
function saveRam(m){
  const bin=atob(m.b64||''); const u=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  download('ram-'+(m.game||'game')+'.bin', new Blob([u],{type:'application/octet-stream'}));
  setStatus('Dumped '+(m.bytes||u.length)+' bytes of RAM.','ok');
}

// ---- Input to per-controller masks. Debug is action-space step-driven. ----
const padByPort=[{}, {}, {}, {}];
document.querySelectorAll('.pad').forEach(pad=>{
  const port=+pad.dataset.port;
  pad.querySelectorAll('.pad-btn').forEach(b=>{
    const bit=+b.dataset.bit; padByPort[port][bit]=b;
    const down=e=>{ e.preventDefault(); if(!drivesPort(port)) return; resumeAudio(); pressed[port].add(bit); recompute(port); };
    const up=()=>{ if(pressed[port].has(bit)){ pressed[port].delete(bit); recompute(port); } };
    b.addEventListener('pointerdown', down);
    b.addEventListener('pointerup', up);
    b.addEventListener('pointerleave', up);
    b.addEventListener('pointercancel', up);
  });
});
function updatePadKeyHints(port, layout){
  const pad=document.querySelector(`.pad[data-port="${port}"]`);
  if(!pad) return;
  pad.querySelectorAll('.pad-btn').forEach(button=>{
    let key=button.querySelector('.k');
    if(!key){
      key=document.createElement('span');
      key.className='k';
      button.appendChild(key);
    }
    key.textContent = layout ? (layout.hints[+button.dataset.bit] || '') : '';
  });
}
function syncPads(){ for(let p=0;p<UI_PORTS;p++){ const m=held[p]; for(const bit in padByPort[p]) padByPort[p][bit].classList.toggle('pressed', (m & +bit)!==0); } }
function recompute(port){
  let m=0; for(const b of pressed[port]) m|=b;
  if(m!==held[port]){ held[port]=m; send({t:'action', port, mask:m}); }
  syncPads();
}
// Route keys by local ownership order, not hardware port number.
function routeKey(e, down){
  const key=e.key && e.key.length===1 ? e.key.toLowerCase() : e.key;
  const ports=localPorts();
  for(let slot=0; slot<ports.length && slot<KEY_LAYOUTS.length; slot++){
    const bit=KEY_LAYOUTS[slot].keys[key];
    if(bit!==undefined){ setKey(ports[slot],bit,down); return true; }
  }
  return false;
}
function setKey(port,bit,down){ if(down) pressed[port].add(bit); else pressed[port].delete(bit); recompute(port); }
document.addEventListener('keydown', e=>{
  // Debug: Space steps all agents; each port has a keyboard row (RL_ACTION_KEYS), action i -> row[i].
  if(mode==='rl'){
    // No e.repeat guard on purpose: holding a key auto-repeats to advance multiple steps.
    if(e.code==='Space' || e.key===' '){
      if(!actionMasks.length) return;
      e.preventDefault(); resumeAudio(); sendStep(); return;
    }
    const ab=$('actBtns'); if(!ab) return;
    const key = e.key && e.key.length===1 ? e.key.toLowerCase() : e.key;
    const rows=ab.querySelectorAll('.agent-actbtns');
    const n=Math.max(1, players);
    for(let p=0; p<n && p<RL_ACTION_KEYS.length; p++){
      const i=RL_ACTION_KEYS[p].indexOf(key);
      if(i>=0 && i<actionMasks.length && rows[p]){
        e.preventDefault(); resumeAudio(); selectAction(p, actionMasks[i], rows[p]); return;
      }
    }
    return;
  }
  if(!controllersActive()) return;
  if(routeKey(e, true)){ e.preventDefault(); resumeAudio(); }
});
document.addEventListener('keyup', e=>{ if(mode==='rl') return; routeKey(e, false); });
window.addEventListener('blur', ()=>{ for(let p=0;p<UI_PORTS;p++){ pressed[p].clear(); recompute(p); } });
const relayoutViewport=()=>{ positionSegInd(); applyPanelLayout(); };
window.addEventListener('resize', relayoutViewport);
if(window.visualViewport){
  window.visualViewport.addEventListener('resize', relayoutViewport);
  window.visualViewport.addEventListener('scroll', relayoutViewport);
}

initViewMenu();
initControlsTabs();
initFloatingPanels();
setMode('play');
requestAnimationFrame(positionSegInd);
connect();
