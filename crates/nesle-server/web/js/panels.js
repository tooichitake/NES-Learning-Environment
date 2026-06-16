import { $, viewportFrame, viewportSize } from './dom.js';

let getMode = () => 'play';
let requestRefreshVis = () => {};

export function configurePanels(options = {}) {
  getMode = options.getMode || getMode;
  requestRefreshVis = options.refreshVis || requestRefreshVis;
}

// ---- Floating panels ----
const PANEL_KEY='nesle-console-panel-layout-v2';
const PANEL_VISIBLE_KEY='nesle-console-panel-visible-v1';
const PANEL_COLLAPSED_KEY='nesle-console-panel-collapsed-v1';
const CONTROLS_TAB_KEY='nesle-console-controls-tab-v1';
const CONTROL_TABS=['session','input','diagnostics'];
let panelLayout={};
let panelVisible={ inspector:true, controls:true };
let panelCollapsed={};
let applyingPanelLayout=false;
let panelZ=30;
let controlsTab='input';
try{ panelLayout=JSON.parse(localStorage.getItem(PANEL_KEY)||'{}')||{}; }catch(e){ panelLayout={}; }
try{ panelVisible={...panelVisible, ...(JSON.parse(localStorage.getItem(PANEL_VISIBLE_KEY)||'{}')||{})}; }catch(e){}
try{ panelCollapsed=JSON.parse(localStorage.getItem(PANEL_COLLAPSED_KEY)||'{}')||{}; }catch(e){ panelCollapsed={}; }
try{
  const savedTab=localStorage.getItem(CONTROLS_TAB_KEY);
  if(CONTROL_TABS.includes(savedTab)) controlsTab=savedTab;
}catch(e){}
export function isPanelVisible(id){
  return panelVisible[id] !== false;
}
function supportsFloating(){
  const vp=viewportSize();
  return vp.w >= 280 && vp.h >= 360;
}
export function canFloatPanels(){
  return supportsFloating();
}
export function panelList(){ return [...document.querySelectorAll('.panel.floatable')]; }
function panelMinWidth(panel){
  if(panel && panel.id==='controlsPanel') return 320;
  if(panel && panel.id==='monitor'){
    const raw=getComputedStyle(panel).getPropertyValue('--monitor-min-w').trim();
    const n=parseFloat(raw);
    return Number.isFinite(n) ? Math.max(260, n) : 260;
  }
  return 260;
}
function panelPreferredWidth(panel){
  if(panel && panel.id==='controlsPanel') return 620;
  if(panel && panel.id==='monitor') return 360;
  return 620;
}
function bringPanelToFront(panel){
  if(!panel || !canFloatPanels() || panel.hidden) return;
  panelZ += 1;
  panel.style.setProperty('--panel-z', String(panelZ));
  for(const p of panelList()) p.classList.toggle('is-key', p===panel);
}
function updatePanelCollapseButton(panel){
  if(!panel) return;
  const id=panel.dataset.panel;
  const btn=document.querySelector(`[data-collapse-panel="${id}"]`);
  if(!btn) return;
  const collapsed=panel.classList.contains('collapsed');
  const name=panel.querySelector('.panel-head h2')?.textContent?.trim() || 'Panel';
  const action=collapsed ? 'Expand' : 'Collapse';
  btn.setAttribute('aria-label', `${action} ${name}`);
  btn.setAttribute('aria-expanded', String(!collapsed));
  btn.title=`${action} ${name}`;
}
function updateAllPanelCollapseButtons(){
  for(const panel of panelList()) updatePanelCollapseButton(panel);
}
function normalizeControlsTab(tab){
  return CONTROL_TABS.includes(tab) ? tab : 'input';
}
export function refreshControlsTabs(){
  controlsTab=normalizeControlsTab(controlsTab);
  document.querySelectorAll('[data-controls-tab]').forEach(btn=>{
    const active=btn.dataset.controlsTab===controlsTab;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
    btn.tabIndex=active ? 0 : -1;
  });
  document.querySelectorAll('[data-controls-module]').forEach(module=>{
    const modeOk=!module.dataset.modes || module.dataset.modes.split(' ').includes(getMode());
    module.hidden=!(modeOk && module.dataset.controlsModule===controlsTab);
  });
}
function setControlsTab(tab){
  controlsTab=normalizeControlsTab(tab);
  localStorage.setItem(CONTROLS_TAB_KEY, controlsTab);
  refreshControlsTabs();
  if(canFloatPanels()){
    const panel=$('controlsPanel');
    if(panel) requestAnimationFrame(()=>storePanel(panel));
  }
}
export function initControlsTabs(){
  const root=$('controlsTabs');
  if(!root) return;
  root.querySelectorAll('[data-controls-tab]').forEach(btn=>{
    btn.addEventListener('click', ()=>setControlsTab(btn.dataset.controlsTab));
  });
  root.addEventListener('keydown', e=>{
    const buttons=[...root.querySelectorAll('[data-controls-tab]')];
    const current=Math.max(0, buttons.findIndex(b=>b.dataset.controlsTab===controlsTab));
    let next=current;
    if(e.key==='ArrowLeft') next=(current+buttons.length-1)%buttons.length;
    else if(e.key==='ArrowRight') next=(current+1)%buttons.length;
    else if(e.key==='Home') next=0;
    else if(e.key==='End') next=buttons.length-1;
    else return;
    setControlsTab(buttons[next].dataset.controlsTab);
    buttons[next].focus();
    e.preventDefault();
  });
  refreshControlsTabs();
}
function setPanelCollapsed(panel, collapsed){
  if(!panel) return;
  const id=panel.dataset.panel;
  const before=panel.getBoundingClientRect();
  panelCollapsed[id]=!!collapsed;
  localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(panelCollapsed));
  panel.classList.toggle('collapsed', !!collapsed);
  if(collapsed){
    const saved=panelLayout[id] || {};
    if(panel.style.height && before.height > 0) panelLayout[id]={ ...saved, h: before.height };
    panel.style.height='';
  }else{
    const saved=panelLayout[id];
    if(saved && saved.h) panel.style.height=saved.h+'px';
  }
  updatePanelCollapseButton(panel);
  if(canFloatPanels()){
    panel.style.left=before.left+'px';
    panel.style.top=before.top+'px';
    requestAnimationFrame(()=>{
      panel.style.left=before.left+'px';
      panel.style.top=before.top+'px';
      storePanel(panel);
    });
  }
}
function screenClearLeft(){
  const dock=document.querySelector('.screen-dock');
  if(!dock) return 330;
  return Math.ceil(dock.getBoundingClientRect().right + 16);
}
function panelWidthBounds(panel, margin, avoidScreen=false){
  const vp=viewportSize();
  const min=panelMinWidth(panel);
  let max=Math.max(min, vp.w - margin*2);
  max=Math.min(max, panelPreferredWidth(panel));
  const clear=screenClearLeft();
  const availableRight=vp.w - clear - margin;
  if(avoidScreen && availableRight >= min) max=Math.min(max, availableRight);
  return { min, max, clear: (avoidScreen && availableRight >= min) ? clear : margin };
}
function panelNaturalHeight(panel){
  if(!panel) return 120;
  const inlineHeight=panel.style.height ? parseFloat(panel.style.height) : NaN;
  if(Number.isFinite(inlineHeight) && inlineHeight > 0) return inlineHeight;
  return Math.max(panel.offsetHeight || 0, panel.scrollHeight || 0, 92);
}
function panelDefaultRects(){
  const margin=16;
  const vp=viewportSize();
  const monitor=$('monitor');
  const controls=$('controlsPanel');
  const screen=document.querySelector('.screen-dock');
  const screenRect=screen ? screen.getBoundingClientRect() : { right: 320, bottom: 360 };
  const statusRect=$('status')?.getBoundingClientRect() || { bottom: 0 };
  const top=Math.max(topLimit(margin), statusRect.bottom + 12);
  const clear=Math.ceil(screenRect.right + 16);
  const gap=16;
  const monitorBounds=panelWidthBounds(monitor, margin, true);
  const controlsBounds=panelWidthBounds(controls, margin, false);
  const monitorW=Math.min(360, Math.max(280, monitorBounds.max));
  const monitorH=panelNaturalHeight(monitor);
  const monitorX=Math.max(margin, Math.min(vp.w - monitorW - margin, vp.w - monitorW - margin));
  const controlsMaxW=Math.min(panelPreferredWidth(controls), Math.max(panelMinWidth(controls), vp.w - margin*2));
  let controlsW=Math.min(controlsMaxW, Math.max(panelMinWidth(controls), controlsBounds.max));
  const availableRight=vp.w - clear - margin;
  const sideBySide=availableRight >= controlsW + monitorW + gap;
  const controlsH=panelNaturalHeight(controls);
  const rects={};
  if(sideBySide){
    rects.inspector={ x: vp.w - monitorW - margin, y: top, w: monitorW };
    rects.controls={ x: Math.max(clear, vp.w - monitorW - controlsW - gap - margin), y: top, w: controlsW };
    return rects;
  }
  const maxControlsBeforeMonitor=Math.max(0, monitorX - margin - gap);
  if(maxControlsBeforeMonitor >= panelMinWidth(controls)){
    controlsW=Math.min(controlsW, maxControlsBeforeMonitor - margin);
  }
  controlsW=Math.max(panelMinWidth(controls), Math.min(controlsW, vp.w - margin*2));
  const preferredControlsY=Math.ceil(screenRect.bottom + 14);
  const controlsY=Math.max(top, Math.min(preferredControlsY, vp.h - Math.min(controlsH, vp.h - margin*2) - margin));
  rects.inspector={ x: monitorX, y: top, w: monitorW };
  rects.controls={ x: margin, y: controlsY, w: controlsW };
  return rects;
}
function panelDefaultRect(panel){
  const id=panel.dataset.panel;
  const rects=panelDefaultRects();
  return rects[id] || rects.controls || { x:16, y:16, w:panelPreferredWidth(panel) };
}
// Minimum Y for any floating panel: the top bar's bottom edge + margin, so panels never hide behind it.
function topLimit(margin){
  const tb=document.querySelector('.topbar');
  const b=tb ? Math.ceil(tb.getBoundingClientRect().bottom) : 0;
  return Math.max(margin, b + margin);
}
function clampPanelRect(rect, panel){
  const margin=12;
  const vp=viewportSize();
  const bounds=panelWidthBounds(panel, margin, false);
  const w=Math.max(bounds.min, Math.min(rect.w || panel.offsetWidth || 320, bounds.max));
  const h=rect.h || panel.offsetHeight || 120;
  const maxX=Math.max(margin, vp.w - w - margin);
  // Top floor is below the top bar, but fall back to `margin` on a cramped viewport so the panel stays visible.
  const fitMaxY=vp.h - Math.min(h, vp.h - margin*2) - margin;
  const minY=topLimit(margin) <= fitMaxY ? topLimit(margin) : margin;
  const maxY=Math.max(minY, fitMaxY);
  return {
    x: Math.max(margin, Math.min(rect.x || margin, maxX)),
    y: Math.max(minY, Math.min(rect.y ?? minY, maxY)),
    w,
    h: rect.h
  };
}
function snapPanel(panel){
  if(!canFloatPanels() || !panel) return;
  const margin=12, snap=18, vp=viewportSize();
  const r=panel.getBoundingClientRect();
  const fitMaxY=vp.h - r.height - margin;
  const minY=topLimit(margin) <= fitMaxY ? topLimit(margin) : margin;
  let x=r.left, y=r.top;
  if(Math.abs(r.left - margin) <= snap) x=margin;
  if(Math.abs((vp.w - r.right) - margin) <= snap) x=Math.max(margin, vp.w - r.width - margin);
  if(Math.abs(r.top - minY) <= snap) y=minY;  // snap to just below the top bar, not the viewport edge
  if(Math.abs((vp.h - r.bottom) - margin) <= snap) y=Math.max(minY, fitMaxY);
  panel.style.left=x+'px';
  panel.style.top=Math.max(minY, y)+'px';
}
function storePanel(panel){
  if(!canFloatPanels() || panel.hidden) return;
  const id=panel.dataset.panel, r=panel.getBoundingClientRect();
  const previous=panelLayout[id] || {};
  panelLayout[id]={
    x:r.left,
    y:r.top,
    w:r.width,
    h:panel.classList.contains('collapsed') ? previous.h : (panel.style.height ? r.height : undefined)
  };
  localStorage.setItem(PANEL_KEY, JSON.stringify(panelLayout));
}
function panelAvailableInCurrentMode(panel){
  const modes=panel.dataset.modes;
  return !modes || modes.split(' ').includes(getMode());
}
export function refreshPanelToggles(){
  const monitor=$('btnToggleMonitor'), controls=$('btnToggleControls');
  if(monitor){
    const available=panelAvailableInCurrentMode($('monitor'));
    monitor.disabled=!available;
    const on=available && panelVisible.inspector!==false;
    monitor.classList.toggle('active', on);
    monitor.setAttribute('aria-pressed', String(on));
    monitor.setAttribute('aria-checked', String(on));
    const state=monitor.querySelector('span:last-child'); if(state) state.textContent=on?'On':'Off';
  }
  if(controls){
    const on=panelVisible.controls!==false;
    controls.classList.toggle('active', on);
    controls.setAttribute('aria-pressed', String(on));
    controls.setAttribute('aria-checked', String(on));
    const state=controls.querySelector('span:last-child'); if(state) state.textContent=on?'On':'Off';
  }
}
export function setPanelVisible(id, visible){
  if(panelVisible[id] === visible) return;
  panelVisible[id]=visible;
  localStorage.setItem(PANEL_VISIBLE_KEY, JSON.stringify(panelVisible));
  requestRefreshVis();
}
window.__nesleRelayoutPanels = () => { if (canFloatPanels()) applyPanelLayout(); };
export function applyPanelLayout(){
  const floating=canFloatPanels();
  document.body.classList.toggle('panels-floating', floating);
  document.body.classList.toggle('panels-docked', !floating);
  applyingPanelLayout=true;
  const defaultRects=floating ? panelDefaultRects() : {};
  for(const panel of panelList()){
    panel.classList.toggle('collapsed', !!panelCollapsed[panel.dataset.panel]);
    if(!floating){
      panel.style.left=''; panel.style.top=''; panel.style.width=''; panel.style.height='';
      panel.style.removeProperty('--panel-z');
      panel.classList.remove('is-key');
      continue;
    }
    const saved=panelLayout[panel.dataset.panel];
    const collapsed=!!panelCollapsed[panel.dataset.panel];
    const rect=clampPanelRect(saved || defaultRects[panel.dataset.panel] || panelDefaultRect(panel), panel);
    panel.style.left=rect.x+'px';
    panel.style.top=rect.y+'px';
    panel.style.width=rect.w+'px';
    panel.style.height=(!collapsed && rect.h) ? (rect.h+'px') : '';
    if(!panel.style.getPropertyValue('--panel-z')) panel.style.setProperty('--panel-z', String(++panelZ));
    updatePanelCollapseButton(panel);
  }
  updateAllPanelCollapseButtons();
  requestAnimationFrame(()=>{ applyingPanelLayout=false; });
}
export function resetPanelLayout(){
  panelLayout={};
  panelZ=30;
  panelVisible={ inspector:true, controls:true };
  panelCollapsed={};
  controlsTab='input';
  localStorage.removeItem('nesle-console-panel-mode-v1');
  localStorage.setItem(PANEL_VISIBLE_KEY, JSON.stringify(panelVisible));
  localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(panelCollapsed));
  localStorage.setItem(CONTROLS_TAB_KEY, controlsTab);
  for(const panel of panelList()){
    panel.style.removeProperty('--panel-z');
    panel.classList.remove('is-key');
  }
  localStorage.removeItem(PANEL_KEY);
  refreshControlsTabs();
  applyPanelLayout();
}
function positionViewMenu(){
  const btn=$('btnViewMenu'), menu=$('viewMenu');
  if(!btn || !menu || menu.hidden) return;
  const r=btn.getBoundingClientRect(), vp=viewportSize(), vf=viewportFrame();
  const mw=menu.offsetWidth || 190, mh=menu.offsetHeight || 120;
  const top=Math.max(8, Math.min(r.bottom + 6, vp.h - mh - 8));
  const left=Math.max(8, Math.min(r.right - mw, vp.w - mw - 8));
  menu.style.top=(vf.top + top)+'px';
  menu.style.left=(vf.left + left)+'px';
}
function closeViewMenu(){
  const btn=$('btnViewMenu'), menu=$('viewMenu');
  if(!btn || !menu) return;
  menu.hidden=true;
  btn.setAttribute('aria-expanded','false');
}
export function initViewMenu(){
  const btn=$('btnViewMenu'), menu=$('viewMenu');
  if(!btn || !menu) return;
  btn.addEventListener('click', e=>{
    const open=menu.hidden;
    menu.hidden=!open;
    btn.setAttribute('aria-expanded', String(open));
    if(open){ positionViewMenu(); requestAnimationFrame(positionViewMenu); }
    e.stopPropagation();
  });
  document.addEventListener('mousedown', e=>{
    if(menu.hidden) return;
    if(!menu.contains(e.target) && !btn.contains(e.target)) closeViewMenu();
  }, true);
  document.addEventListener('keydown', e=>{
    if(e.key==='Escape') closeViewMenu();
  }, true);
  window.addEventListener('resize', positionViewMenu);
  if(window.visualViewport){
    window.visualViewport.addEventListener('resize', positionViewMenu);
    window.visualViewport.addEventListener('scroll', positionViewMenu);
  }
}
export function initFloatingPanels(){
  let drag=null, saveTimer=null;
  const scheduleSave=panel=>{
    clearTimeout(saveTimer);
    saveTimer=setTimeout(()=>storePanel(panel), 150);
  };
  const moveDrag=e=>{
    if(!drag) return;
    const vp=viewportSize();
    const panel=drag.panel, w=panel.offsetWidth, h=panel.offsetHeight, margin=12;
    const fitMaxY=vp.h-h-margin;
    const minY=topLimit(margin) <= fitMaxY ? topLimit(margin) : margin;
    const x=Math.max(margin, Math.min(e.clientX-drag.dx, vp.w-w-margin));
    const y=Math.max(minY, Math.min(e.clientY-drag.dy, fitMaxY));
    panel.style.left=x+'px'; panel.style.top=y+'px';
  };
  const endDrag=()=>{
    if(!drag) return;
    drag.panel.classList.remove('dragging');
    snapPanel(drag.panel);
    storePanel(drag.panel);
    drag=null;
    window.removeEventListener('pointermove', moveDrag);
    window.removeEventListener('pointerup', endDrag);
    window.removeEventListener('pointercancel', endDrag);
    window.removeEventListener('mousemove', moveDrag);
    window.removeEventListener('mouseup', endDrag);
  };
  for(const panel of panelList()){
    panel.addEventListener('pointerdown', ()=>bringPanelToFront(panel), {capture:true});
    panel.addEventListener('focusin', ()=>bringPanelToFront(panel));
    const handle=panel.querySelector('[data-panel-handle]');
    if(handle){
      handle.title='Drag to move. Use the toolbar button to collapse.';
      const startDrag=e=>{
        if(!supportsFloating() || e.button!==0 || e.target.closest('button,input,select,label')) return;
        const r=panel.getBoundingClientRect();
        if(drag) endDrag();
        bringPanelToFront(panel);
        drag={ panel, dx:e.clientX-r.left, dy:e.clientY-r.top };
        panel.classList.add('dragging');
        window.addEventListener('pointermove', moveDrag);
        window.addEventListener('pointerup', endDrag);
        window.addEventListener('pointercancel', endDrag);
        window.addEventListener('mousemove', moveDrag);
        window.addEventListener('mouseup', endDrag);
        e.preventDefault();
      };
      handle.addEventListener('pointerdown', startDrag);
      handle.addEventListener('mousedown', startDrag);
      const toggleCollapse=e=>{
        if(e.target.closest('button,input,select,label')) return;
        setPanelCollapsed(panel, !panel.classList.contains('collapsed'));
        e.preventDefault();
        e.stopPropagation();
      };
      const collapseOnDoublePress=e=>{ if(e.detail>=2) toggleCollapse(e); };
      handle.addEventListener('pointerdown', collapseOnDoublePress, {capture:true});
      handle.addEventListener('mousedown', collapseOnDoublePress, {capture:true});
      handle.addEventListener('dblclick', toggleCollapse);
      handle.querySelector('h2')?.addEventListener('dblclick', toggleCollapse);
    }
    if(window.ResizeObserver){
      const ro=new ResizeObserver(()=>{ if(canFloatPanels() && !applyingPanelLayout && !panel.hidden && !panel.classList.contains('dragging')) scheduleSave(panel); });
      ro.observe(panel);
    }
  }
  const btn=$('btnResetLayout');
  if(btn) btn.addEventListener('click', resetPanelLayout);
  const menuBtn=$('btnResetLayoutMenu');
  if(menuBtn) menuBtn.addEventListener('click', ()=>{ resetPanelLayout(); closeViewMenu(); });
  document.querySelectorAll('[data-hide-panel]').forEach(btn=>{
    btn.addEventListener('click', ()=>setPanelVisible(btn.dataset.hidePanel, false));
  });
  document.querySelectorAll('[data-collapse-panel]').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const panel=panelList().find(p=>p.dataset.panel===btn.dataset.collapsePanel);
      if(panel) setPanelCollapsed(panel, !panel.classList.contains('collapsed'));
    });
  });
  const monitorToggle=$('btnToggleMonitor'), controlsToggle=$('btnToggleControls');
  if(monitorToggle) monitorToggle.addEventListener('click', ()=>setPanelVisible('inspector', !(panelVisible.inspector!==false)));
  if(controlsToggle) controlsToggle.addEventListener('click', ()=>setPanelVisible('controls', !(panelVisible.controls!==false)));
  applyPanelLayout();
}
