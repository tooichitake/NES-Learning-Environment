export const $ = id => document.getElementById(id);

export function viewportSize() {
  const vv = window.visualViewport;
  return {
    w: Math.max(1, Math.floor((vv && vv.width) || document.documentElement.clientWidth || window.innerWidth || 1)),
    h: Math.max(1, Math.floor((vv && vv.height) || document.documentElement.clientHeight || window.innerHeight || 1)),
  };
}

export function viewportFrame() {
  const vv = window.visualViewport;
  const size = viewportSize();
  return {
    ...size,
    left: Math.floor((vv && vv.offsetLeft) || 0),
    top: Math.floor((vv && vv.offsetTop) || 0),
    right: Math.floor((vv && vv.offsetLeft) || 0) + size.w,
    bottom: Math.floor((vv && vv.offsetTop) || 0) + size.h,
  };
}
