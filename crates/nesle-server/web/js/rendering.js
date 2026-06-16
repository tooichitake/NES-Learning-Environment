const scratch = document.createElement('canvas');
const sctx = scratch.getContext('2d');

export function setMonitorMinWidth(px) {
  const monitor = document.getElementById('monitor');
  if (!monitor) return;
  const next = Math.max(260, Math.min(340, Math.ceil(px))) + 'px';
  if (monitor.style.getPropertyValue('--monitor-min-w') === next) return;
  monitor.style.setProperty('--monitor-min-w', next);
  window.__nesleRelayoutPanels?.();
}

export function blit(dst, dctx, data, w, h, channels) {
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w;
    scratch.height = h;
  }
  const img = sctx.createImageData(w, h);
  const out = img.data;
  if (channels === 3) {
    for (let i = 0, n = w * h; i < n; i++) {
      const i3 = i * 3;
      const i4 = i * 4;
      out[i4] = data[i3];
      out[i4 + 1] = data[i3 + 1];
      out[i4 + 2] = data[i3 + 2];
      out[i4 + 3] = 255;
    }
  } else {
    for (let i = 0, n = w * h; i < n; i++) {
      const value = data[i];
      const i4 = i * 4;
      out[i4] = value;
      out[i4 + 1] = value;
      out[i4 + 2] = value;
      out[i4 + 3] = 255;
    }
  }
  sctx.putImageData(img, 0, 0);
  if (dst.width !== w || dst.height !== h) {
    dst.width = w;
    dst.height = h;
  }
  if (dst.id !== 'native') {
    dst.style.width = w + 'px';
    dst.style.height = h + 'px';
    if (dst.id === 'obs') setMonitorMinWidth(w + 54);
  }
  dctx.imageSmoothingEnabled = false;
  dctx.drawImage(scratch, 0, 0);
}
