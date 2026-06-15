use std::cell::RefCell;

use wide::{i32x8, u8x16};

use crate::constants::{GRAY_FRAME_LEN, NES_HEIGHT, NES_WIDTH};

const NATIVE_W: usize = NES_WIDTH;
const NATIVE_H: usize = NES_HEIGHT;
const NATIVE_N: usize = GRAY_FRAME_LEN;

/// Per-axis INTER_AREA resampling plan with **exact-integer** weights.
///
/// OpenCV `INTER_AREA` weights are the fractional overlaps of each source pixel
/// with the destination pixel's real-valued footprint `[d*src/dst,(d+1)*src/dst)`.
/// Those overlaps are rationals with denominator `dst`, so scaling each one by
/// `dst` yields an **exact integer** with no float and no quantization:
///
/// ```text
/// w_int(i,d) = min((i+1)*dst, (d+1)*src) - max(i*dst, d*src)   (clamped >= 0)
/// ```
///
/// Weights are stored row-major `[out*k + tap]`, zero-padded to a fixed support
/// `k = max overlap count`. `sums[d]` is the per-output weight sum (the axis
/// normaliser `S`); the destination value is `round( (Σ wx·wy·src) / (Sx·Sy) )`,
/// accumulated with the products of the two axes' integer weights -- algebraically
/// identical to the dense 2-D `INTER_AREA` average, just regrouped into two
/// separable passes (vertical then horizontal).
#[derive(Debug, Clone)]
struct AxisPlan {
    /// Padded support width (max number of source taps over all outputs).
    k: usize,
    /// First source index contributing to each output (len `dst`).
    starts: Vec<usize>,
    /// Integer overlap weights, row-major `[out*k + tap]`, zero-padded (len `dst*k`).
    weights: Vec<i32>,
    /// Per-output weight sum (the axis normaliser `S`, len `dst`).
    sums: Vec<i32>,
}

impl AxisPlan {
    fn new(src: usize, dst: usize) -> Self {
        let mut starts = vec![0usize; dst];
        let mut sums = vec![0i32; dst];
        let mut raw: Vec<Vec<i32>> = Vec::with_capacity(dst);
        let mut k = 0usize;
        for (d, start_slot) in starts.iter_mut().enumerate() {
            // Footprint [d*src, (d+1)*src) scaled by `dst`, so overlaps are exact integers.
            let lo = d * src;
            let hi = (d + 1) * src;
            let i0 = lo / dst; // floor(d*src/dst)
            let i1 = (hi.div_ceil(dst)).min(src); // ceil((d+1)*src/dst), clamped to image
            let mut ws = Vec::with_capacity(i1 - i0);
            let mut s = 0i32;
            for i in i0..i1 {
                let left = (i * dst).max(lo);
                let right = ((i + 1) * dst).min(hi);
                let w = (right as i64 - left as i64).max(0) as i32;
                ws.push(w);
                s += w;
            }
            *start_slot = i0;
            sums[d] = s;
            k = k.max(ws.len());
            raw.push(ws);
        }
        let k = k.max(1);
        let mut weights = vec![0i32; dst * k];
        for (d, ws) in raw.iter().enumerate() {
            weights[d * k..d * k + ws.len()].copy_from_slice(ws);
        }
        Self {
            k,
            starts,
            weights,
            sums,
        }
    }
}

/// Precomputed INTER_AREA resize plan for a fixed source/destination shape.
///
/// Implemented as a **separable exact-integer convolution**: a vertical
/// (row-combine) pass that reduces height `src_h -> dst_h` while keeping the full
/// `src_w` width, followed by a horizontal (column-combine) pass that reduces
/// width `src_w -> dst_w` and applies the single INTER_AREA rounding. The
/// vertical pass reads **contiguous source rows**, so it is the one that is SIMD
/// vectorized (`wide::i32x8`, → AVX2/SSE2/NEON); it is also where most of the
/// work is (it touches every source row), which is why this order -- not the
/// reverse -- is the fast one. Because the 2-D `INTER_AREA` weight is the product
/// `wx*wy`, the separable form is byte-exact to the dense 2-D integer average.
///
/// Holds reusable scratch (`inter`, `pooled`) so per-step resizing never
/// allocates. The scratch lives behind `RefCell` so it can be mutated through a
/// shared `&ResizePlan`: each `ObsWindow` owns its own plan and is stepped by a
/// single thread (in the worker pool each env is owned exclusively by one worker),
/// so no `Sync` / locking is required.
#[derive(Debug, Clone)]
pub struct ResizePlan {
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    x: AxisPlan,
    y: AxisPlan,
    /// Vertical-pass intermediate, row-major `inter[dy*src_w + i]`, len `src_w*dst_h`.
    inter: RefCell<Vec<i32>>,
    /// Max-pool scratch (elementwise max of two source frames); sized lazily.
    pooled: RefCell<Vec<u8>>,
}

impl ResizePlan {
    pub fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        assert!(src_w > 0 && src_h > 0, "source dimensions must be positive");
        assert!(
            dst_w > 0 && dst_h > 0,
            "destination dimensions must be positive"
        );
        Self {
            src_w,
            src_h,
            dst_w,
            dst_h,
            x: AxisPlan::new(src_w, dst_w),
            y: AxisPlan::new(src_h, dst_h),
            inter: RefCell::new(vec![0i32; src_w * dst_h]),
            pooled: RefCell::new(Vec::new()),
        }
    }

    pub fn output_len(&self) -> usize {
        self.dst_w * self.dst_h
    }

    fn is_identity(&self) -> bool {
        self.dst_w == self.src_w && self.dst_h == self.src_h
    }
}

/// Vertical (row-combine) pass: reduce height `src_h -> dst_h`, keeping full width
/// `src_w`, into the unrounded i32 intermediate `inter[dy*src_w + i]`. Reads one
/// channel of `src` (element stride `stride`, byte offset `off`). For the
/// single-channel grayscale case (`stride == 1`) each source row is contiguous,
/// so the multiply-accumulate is vectorized with `wide::i32x8`. Per-cell value is
/// `Σ_t wy·src ≤ Sy·255 = src_h·255`, well within i32 for any real frame.
fn vertical_pass(src: &[u8], stride: usize, off: usize, plan: &ResizePlan, inter: &mut [i32]) {
    let y = &plan.y;
    let (src_w, src_h, dst_h) = (plan.src_w, plan.src_h, plan.dst_h);
    let max_row = src_h - 1;
    let ky = y.k;
    for dy in 0..dst_h {
        let start = y.starts[dy];
        let wbase = dy * y.k;
        let irow = &mut inter[dy * src_w..(dy + 1) * src_w];
        if stride == 1 && off == 0 {
            // Contiguous rows -> branch-free fixed-length tap loop (zero-padded taps mul by 0), i32x8 widening MAC.
            let mut i = 0;
            while i + 8 <= src_w {
                let mut acc = i32x8::splat(0);
                for t in 0..ky {
                    let w = y.weights[wbase + t];
                    let sbase = (start + t).min(max_row) * src_w + i;
                    let s = &src[sbase..sbase + 8];
                    let sv = i32x8::from([
                        s[0] as i32,
                        s[1] as i32,
                        s[2] as i32,
                        s[3] as i32,
                        s[4] as i32,
                        s[5] as i32,
                        s[6] as i32,
                        s[7] as i32,
                    ]);
                    acc += i32x8::splat(w) * sv;
                }
                irow[i..i + 8].copy_from_slice(&acc.to_array());
                i += 8;
            }
            while i < src_w {
                let mut a = 0i32;
                for t in 0..ky {
                    let row = (start + t).min(max_row);
                    a += y.weights[wbase + t] * src[row * src_w + i] as i32;
                }
                irow[i] = a;
                i += 1;
            }
        } else {
            for (i, slot) in irow.iter_mut().enumerate() {
                let mut a = 0i32;
                for t in 0..ky {
                    let row = (start + t).min(max_row);
                    a += y.weights[wbase + t] * src[(row * src_w + i) * stride + off] as i32;
                }
                *slot = a;
            }
        }
    }
}

/// Horizontal (column-combine) pass over the i32 intermediate, reducing width
/// `src_w -> dst_w` and applying the single INTER_AREA rounding. Writes one
/// channel of `dst` (element stride `stride`, offset `off`). `out = round( acc /
/// (Sx·Sy) )` via integer round-half-up `(2*acc + area) / (2*area)`, which equals
/// f64 `.round()` for the non-negative `acc`, `area` here. This is the gather pass
/// (each output reads a sliding source window) but runs on the already
/// height-reduced intermediate, so it is the smaller of the two passes.
fn horizontal_pass_round(
    inter: &[i32],
    plan: &ResizePlan,
    stride: usize,
    off: usize,
    dst: &mut [u8],
) {
    let (x, y) = (&plan.x, &plan.y);
    let (src_w, dst_w, dst_h) = (plan.src_w, plan.dst_w, plan.dst_h);
    let max_idx = src_w - 1;
    for dy in 0..dst_h {
        let sy = y.sums[dy] as i64;
        let ibase = dy * src_w;
        for dx in 0..dst_w {
            let start = x.starts[dx];
            let wbase = dx * x.k;
            let area = x.sums[dx] as i64 * sy;
            let mut acc = 0i64;
            for t in 0..x.k {
                let w = x.weights[wbase + t] as i64;
                let idx = (start + t).min(max_idx);
                acc += w * inter[ibase + idx] as i64;
            }
            dst[(dy * dst_w + dx) * stride + off] = ((2 * acc + area) / (2 * area)) as u8;
        }
    }
}

/// Area-weighted downscale of a single-channel (grayscale) frame, matching
/// OpenCV `INTER_AREA` (the interpolation ALE / DeepMind use to make the 84x84
/// Atari observation): each destination pixel is the area-weighted average of the
/// source pixels its real-valued footprint `[dx*sx,(dx+1)*sx) x [dy*sy,(dy+1)*sy)`
/// overlaps. Returns the source unchanged when the destination size equals the
/// source size.
///
/// `src` is row-major, length `src_w * src_h`. Panics if it is not, or if the
/// destination dimensions are zero (callers validate user input upstream).
pub fn resize_area_gray(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h, "src len must equal src_w*src_h");
    assert!(
        dst_w > 0 && dst_h > 0,
        "destination dimensions must be positive"
    );
    if dst_w == src_w && dst_h == src_h {
        return src.to_vec();
    }
    let plan = ResizePlan::new(src_w, src_h, dst_w, dst_h);
    let mut dst = vec![0u8; dst_w * dst_h];
    resize_area_gray_into(src, &plan, &mut dst);
    dst
}

/// Resize with a precomputed [`ResizePlan`], writing into caller-owned storage.
pub fn resize_area_gray_into(src: &[u8], plan: &ResizePlan, dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        plan.src_w * plan.src_h,
        "src len must equal plan source dimensions"
    );
    assert_eq!(
        dst.len(),
        plan.output_len(),
        "dst len must equal plan destination dimensions"
    );
    if plan.is_identity() {
        dst.copy_from_slice(src);
        return;
    }
    let mut inter = plan.inter.borrow_mut();
    vertical_pass(src, 1, 0, plan, &mut inter);
    horizontal_pass_round(&inter, plan, 1, 0, dst);
}

/// Max-pool two source frames (the ALE flicker fix) then INTER_AREA resize. The
/// elementwise `frame0.max(frame1)` is applied at full source resolution into a
/// reusable scratch buffer, then the normal separable resize runs on the pooled
/// frame -- term-for-term identical to folding `max` inside the weighted sum,
/// since `max` is per source pixel and independent of the linear resize weights.
fn resize_area_gray_max_into(frame0: &[u8], frame1: &[u8], plan: &ResizePlan, dst: &mut [u8]) {
    assert_eq!(
        frame0.len(),
        plan.src_w * plan.src_h,
        "frame0 len must equal plan source dimensions"
    );
    assert_eq!(
        frame1.len(),
        plan.src_w * plan.src_h,
        "frame1 len must equal plan source dimensions"
    );
    assert_eq!(
        dst.len(),
        plan.output_len(),
        "dst len must equal plan destination dimensions"
    );
    if plan.is_identity() {
        for (d, (&a, &b)) in dst.iter_mut().zip(frame0.iter().zip(frame1)) {
            *d = a.max(b);
        }
        return;
    }
    let mut pooled = plan.pooled.borrow_mut();
    if pooled.len() != frame0.len() {
        pooled.resize(frame0.len(), 0);
    }
    // Elementwise max (ALE flicker fix) vectorizes portably with `wide::u8x16`; byte-identical to scalar max.
    let n = frame0.len();
    let mut i = 0;
    while i + 16 <= n {
        let a = u8x16::from(<[u8; 16]>::try_from(&frame0[i..i + 16]).unwrap());
        let b = u8x16::from(<[u8; 16]>::try_from(&frame1[i..i + 16]).unwrap());
        pooled[i..i + 16].copy_from_slice(&a.max(b).to_array());
        i += 16;
    }
    for j in i..n {
        pooled[j] = frame0[j].max(frame1[j]);
    }
    let mut inter = plan.inter.borrow_mut();
    vertical_pass(&pooled, 1, 0, plan, &mut inter);
    horizontal_pass_round(&inter, plan, 1, 0, dst);
}

/// Compute one observation from the most-recent native grayscale frame(s):
/// element-wise max-pool with the previous frame when `prev` is `Some` (the ALE
/// flicker fix), then INTER_AREA resize to `screen_size` x `screen_size`. Frames
/// are row-major `256*240`. This is the pixel math both the Python wrapper and the
/// server share -- no per-frontend reimplementation.
pub fn compute_obs(frame0: &[u8], prev: Option<&[u8]>, screen_size: usize) -> Vec<u8> {
    let plan = ResizePlan::new(NATIVE_W, NATIVE_H, screen_size, screen_size);
    let mut obs = vec![0u8; plan.output_len()];
    compute_obs_into(frame0, prev, &plan, &mut obs);
    obs
}

pub fn compute_obs_into(frame0: &[u8], prev: Option<&[u8]>, plan: &ResizePlan, obs: &mut [u8]) {
    assert_eq!(frame0.len(), NATIVE_N, "frame0 must be a native NES frame");
    match prev {
        Some(f1) => {
            assert_eq!(f1.len(), NATIVE_N, "prev must be a native NES frame");
            resize_area_gray_max_into(frame0, f1, plan, obs);
        }
        None => resize_area_gray_into(frame0, plan, obs),
    }
}

/// Phase C SIMD resize path (opt-in via `simd-preprocess` feature).
///
/// Wraps [`fast_image_resize`](https://github.com/Cykooz/fast_image_resize) which
/// runtime-dispatches to AVX2/SSE4.1 on x86_64 and NEON on aarch64. The
/// `Resizer` is reusable across calls and caches the per-(src,dst) shape
/// coefficient tables internally; constructing it costs ~tens of microseconds,
/// which is amortized over thousands of frames in normal use.
///
/// **Output is NOT byte-equivalent to [`resize_area_gray_into`]**: the SIMD
/// filter is `ResizeAlg::Convolution(FilterType::Box)`, whose weights differ from
/// OpenCV INTER_AREA for non-integer downscale ratios. The default
/// [`resize_area_gray_into`] is now itself SIMD-accelerated (via `wide`) and
/// byte-exact to integer INTER_AREA, so this opt-in path is retained only for one
/// release as a comparison baseline and will be removed. See the
/// `simd_resize_matches_scalar_within_tolerance` test for the measured worst-case
/// per-pixel divergence on representative sizes.
#[cfg(feature = "simd-preprocess")]
pub struct SimdResizer {
    inner: fast_image_resize::Resizer,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    // Mutable source copy: `Image::from_slice_u8` requires &mut even for inputs (no read-only constructor).
    src_scratch: Vec<u8>,
}

#[cfg(feature = "simd-preprocess")]
impl SimdResizer {
    pub fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        assert!(src_w > 0 && src_h > 0, "source dimensions must be positive");
        assert!(
            dst_w > 0 && dst_h > 0,
            "destination dimensions must be positive"
        );
        Self {
            inner: fast_image_resize::Resizer::new(),
            src_w,
            src_h,
            dst_w,
            dst_h,
            src_scratch: vec![0u8; src_w * src_h],
        }
    }

    pub fn output_len(&self) -> usize {
        self.dst_w * self.dst_h
    }

    /// Resize one grayscale (U8, single channel) frame into caller-owned storage.
    /// `src` must be `src_w * src_h` bytes; `dst` must be `dst_w * dst_h` bytes.
    pub fn resize_gray_into(&mut self, src: &[u8], dst: &mut [u8]) {
        use fast_image_resize::images::Image;
        use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions};
        assert_eq!(src.len(), self.src_w * self.src_h, "src len mismatch");
        assert_eq!(dst.len(), self.dst_w * self.dst_h, "dst len mismatch");
        if self.dst_w == self.src_w && self.dst_h == self.src_h {
            dst.copy_from_slice(src);
            return;
        }
        self.src_scratch.copy_from_slice(src);
        let src_img = Image::from_slice_u8(
            self.src_w as u32,
            self.src_h as u32,
            &mut self.src_scratch,
            PixelType::U8,
        )
        .expect("src image construction");
        let mut dst_img =
            Image::from_slice_u8(self.dst_w as u32, self.dst_h as u32, dst, PixelType::U8)
                .expect("dst image construction");
        let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Box));
        self.inner
            .resize(&src_img, &mut dst_img, &opts)
            .expect("SIMD resize");
    }
}

/// RGB (3-channel interleaved) INTER_AREA downscale, byte-exact to the grayscale
/// path applied per channel. Reuses the plan's i32 intermediate one channel at a
/// time (RGB is not the default obs, so the per-channel strided reads -- which
/// cannot use the contiguous SIMD vertical pass -- are acceptable).
pub fn resize_area_rgb_into(src: &[u8], plan: &ResizePlan, dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        plan.src_w * plan.src_h * 3,
        "src len must equal plan source dimensions times 3"
    );
    assert_eq!(
        dst.len(),
        plan.output_len() * 3,
        "dst len must equal plan destination dimensions times 3"
    );
    if plan.is_identity() {
        dst.copy_from_slice(src);
        return;
    }
    let mut inter = plan.inter.borrow_mut();
    for c in 0..3 {
        vertical_pass(src, 3, c, plan, &mut inter);
        horizontal_pass_round(&inter, plan, 3, c, dst);
    }
}
