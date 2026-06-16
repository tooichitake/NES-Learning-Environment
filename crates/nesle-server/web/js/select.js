import { viewportFrame, viewportSize } from './dom.js';

export function enhanceSelect(sel) {
  if (sel.dataset.enhanced) return;
  sel.dataset.enhanced = '1';
  const wrap = document.createElement('span');
  wrap.className = 'xsel';
  const trig = document.createElement('button');
  trig.type = 'button';
  trig.className = 'xsel-trigger';
  trig.setAttribute('aria-haspopup', 'listbox');
  trig.setAttribute('aria-expanded', 'false');
  const caret = '<svg viewBox="0 0 12 12" width="11" height="11"><path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const label = value => {
    const option = [...sel.options].find(item => item.value === value);
    return option ? option.textContent : value;
  };
  const sync = () => {
    trig.innerHTML = '';
    const text = document.createElement('span');
    text.textContent = label(sel.value);
    trig.appendChild(text);
    trig.insertAdjacentHTML('beforeend', caret);
  };
  sel.parentNode.insertBefore(wrap, sel);
  wrap.appendChild(trig);
  wrap.appendChild(sel);
  let menu = null;
  let pointerStartedInMenu = false;
  const onEsc = event => {
    if (menu && event.key === 'Escape') {
      close();
      trig.focus();
      event.preventDefault();
    }
  };
  const close = () => {
    if (!menu) return;
    menu.remove();
    menu = null;
    pointerStartedInMenu = false;
    trig.setAttribute('aria-expanded', 'false');
    document.removeEventListener('mousedown', onPointerStart, true);
    document.removeEventListener('click', onDoc, true);
    document.removeEventListener('keydown', onEsc, true);
    document.removeEventListener('wheel', onWheel, { capture: true });
    window.removeEventListener('resize', close);
  };
  const eventInside = (event, el) => {
    if (!el) return false;
    const path = event.composedPath ? event.composedPath() : [];
    if (el.contains(event.target) || path.includes(el)) return true;
    if (Number.isFinite(event.clientX) && Number.isFinite(event.clientY)) {
      const rect = el.getBoundingClientRect();
      return event.clientX >= rect.left - 1 && event.clientX <= rect.right + 1
        && event.clientY >= rect.top - 1 && event.clientY <= rect.bottom + 1;
    }
    return false;
  };
  const onPointerStart = event => {
    pointerStartedInMenu = eventInside(event, menu);
  };
  const onDoc = event => {
    if (pointerStartedInMenu) {
      pointerStartedInMenu = false;
      return;
    }
    if (menu && !eventInside(event, menu) && !eventInside(event, wrap)) close();
  };
  const onWheel = event => {
    if (!menu || !eventInside(event, menu)) return;
    const before = menu.scrollTop;
    menu.scrollTop += event.deltaY;
    if (menu.scrollTop !== before) event.preventDefault();
    event.stopPropagation();
  };
  const choose = option => {
    if (sel.value !== option.value) {
      sel.value = option.value;
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    }
    sync();
    close();
    trig.focus();
  };
  const open = () => {
    if (menu) {
      close();
      return;
    }
    menu = document.createElement('div');
    menu.className = 'xsel-menu glass';
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-label', (sel.closest('label')?.textContent || 'Options').trim());
    [...sel.options].forEach(option => {
      const item = document.createElement('div');
      item.className = 'xsel-item' + (option.value === sel.value ? ' sel' : '');
      item.textContent = option.textContent;
      item.tabIndex = -1;
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(option.value === sel.value));
      item.addEventListener('click', () => choose(option));
      item.addEventListener('keydown', event => {
        const items = [...menu.querySelectorAll('.xsel-item')];
        const index = Math.max(0, items.indexOf(document.activeElement));
        if (event.key === 'Enter' || event.key === ' ') {
          choose(option);
          event.preventDefault();
        } else if (event.key === 'ArrowDown') {
          items[Math.min(items.length - 1, index + 1)].focus();
          event.preventDefault();
        } else if (event.key === 'ArrowUp') {
          items[Math.max(0, index - 1)].focus();
          event.preventDefault();
        }
      });
      menu.appendChild(item);
    });
    document.body.appendChild(menu);
    const r = trig.getBoundingClientRect();
    const vp = viewportSize();
    const vf = viewportFrame();
    const menuMaxW = Math.max(170, vp.w - 16);
    const menuMaxH = Math.max(96, vp.h - 16);
    menu.style.minWidth = Math.min(r.width, menuMaxW) + 'px';
    menu.style.maxWidth = menuMaxW + 'px';
    menu.style.maxHeight = Math.min(menuMaxH, Math.max(96, Math.floor(vp.h * 0.6))) + 'px';
    trig.setAttribute('aria-expanded', 'true');
    const mh = Math.min(menu.offsetHeight, menuMaxH);
    const mw = Math.min(menu.offsetWidth, menuMaxW);
    const below = vp.h - r.bottom - 8;
    const above = r.top - 8;
    let top = below >= mh || below >= above ? r.bottom + 6 : r.top - 6 - mh;
    top = Math.max(8, Math.min(top, vp.h - mh - 8));
    let left = Math.max(8, Math.min(r.left, vp.w - mw - 8));
    menu.style.top = vf.top + top + 'px';
    menu.style.left = vf.left + left + 'px';
    document.addEventListener('mousedown', onPointerStart, true);
    document.addEventListener('click', onDoc, true);
    document.addEventListener('keydown', onEsc, true);
    document.addEventListener('wheel', onWheel, { capture: true, passive: false });
    window.addEventListener('resize', close);
    requestAnimationFrame(() => (menu.querySelector('.xsel-item.sel') || menu.querySelector('.xsel-item'))?.focus());
  };
  trig.addEventListener('click', open);
  trig.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
      open();
      event.preventDefault();
    }
  });
  sel.addEventListener('change', sync);
  sel._sync = sync;
  sync();
  return sync;
}
