//! Shared NESLE observation preprocessing.

mod config;
mod pipeline;
mod resize;
mod window;

pub use config::{ObsConfig, ObsKind, ObsShape, RenderPolicy, RewardClip};
pub use pipeline::{ObsPipeline, ObsStepMeta};
#[cfg(feature = "simd-preprocess")]
pub use resize::SimdResizer;
pub use resize::{
    compute_obs, compute_obs_into, resize_area_gray, resize_area_gray_into, resize_area_rgb_into,
    ResizePlan,
};
pub use window::{FrameSample, ObsWindow, ObsWindowStep};

#[cfg(test)]
const NATIVE_W: usize = crate::constants::NES_WIDTH;
#[cfg(test)]
const NATIVE_H: usize = crate::constants::NES_HEIGHT;
#[cfg(test)]
const NATIVE_N: usize = crate::constants::GRAY_FRAME_LEN;

#[cfg(test)]
mod tests {
    use super::*;

    fn resize_area_gray_scalar_reference(
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        assert_eq!(src.len(), src_w * src_h);
        let sx = src_w as f64 / dst_w as f64;
        let sy = src_h as f64 / dst_h as f64;
        let mut dst = vec![0u8; dst_w * dst_h];
        for dy in 0..dst_h {
            let y0 = dy as f64 * sy;
            let y1 = y0 + sy;
            let iy0 = y0.floor() as usize;
            let iy1 = (y1.ceil() as usize).min(src_h);
            for dx in 0..dst_w {
                let x0 = dx as f64 * sx;
                let x1 = x0 + sx;
                let ix0 = x0.floor() as usize;
                let ix1 = (x1.ceil() as usize).min(src_w);
                let mut acc = 0.0f64;
                let mut area = 0.0f64;
                for yy in iy0..iy1 {
                    let wy = ((yy + 1) as f64).min(y1) - (yy as f64).max(y0);
                    if wy <= 0.0 {
                        continue;
                    }
                    let row = yy * src_w;
                    for xx in ix0..ix1 {
                        let wx = ((xx + 1) as f64).min(x1) - (xx as f64).max(x0);
                        if wx <= 0.0 {
                            continue;
                        }
                        let w = wx * wy;
                        acc += w * src[row + xx] as f64;
                        area += w;
                    }
                }
                dst[dy * dst_w + dx] = (acc / area).round() as u8;
            }
        }
        dst
    }

    /// Exact-integer 2-D INTER_AREA oracle: the same area-overlap weights as the
    /// production path but accumulated densely (non-separable) with integer
    /// round-half-up. Because integer addition is associative, the production
    /// separable path must equal this byte-for-byte -- this is the correctness
    /// oracle for the SIMD rewrite (the f64 reference above only agrees to within
    /// 1, since `sx = src/dst` is not dyadic and mis-rounds exact-half ties).
    fn resize_area_gray_int_reference(
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        let axis = |s: usize, d: usize, o: usize| -> (usize, usize) {
            let lo = o * s;
            let hi = (o + 1) * s;
            (lo / d, hi.div_ceil(d).min(s))
        };
        let w = |s: usize, d: usize, o: usize, i: usize| -> i64 {
            let lo = o * s;
            let hi = (o + 1) * s;
            (((i + 1) * d).min(hi) as i64 - (i * d).max(lo) as i64).max(0)
        };
        let mut dst = vec![0u8; dst_w * dst_h];
        for dy in 0..dst_h {
            let (j0, j1) = axis(src_h, dst_h, dy);
            let sy: i64 = (j0..j1).map(|j| w(src_h, dst_h, dy, j)).sum();
            for dx in 0..dst_w {
                let (i0, i1) = axis(src_w, dst_w, dx);
                let sx: i64 = (i0..i1).map(|i| w(src_w, dst_w, dx, i)).sum();
                let mut acc = 0i64;
                for j in j0..j1 {
                    let wy = w(src_h, dst_h, dy, j);
                    for i in i0..i1 {
                        acc += w(src_w, dst_w, dx, i) * wy * src[j * src_w + i] as i64;
                    }
                }
                let area = sx * sy;
                dst[dy * dst_w + dx] = ((2 * acc + area) / (2 * area)) as u8;
            }
        }
        dst
    }

    #[test]
    fn compute_obs_no_prev_equals_plain_resize() {
        let f = vec![100u8; NATIVE_N];
        assert_eq!(
            compute_obs(&f, None, 8),
            resize_area_gray(&f, NATIVE_W, NATIVE_H, 8, 8)
        );
    }

    #[test]
    fn compute_obs_maxpools_then_resizes() {
        let mut a = vec![10u8; NATIVE_N];
        let mut b = vec![20u8; NATIVE_N];
        a[0] = 200;
        b[1] = 150;
        let manual: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
        assert_eq!(
            compute_obs(&a, Some(&b), 84),
            resize_area_gray_int_reference(&manual, NATIVE_W, NATIVE_H, 84, 84)
        );
    }

    fn max_abs_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as i32 - y as i32).abs())
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn separable_resize_is_byte_exact_to_integer_inter_area() {
        // The separable path must equal the dense integer INTER_AREA oracle byte-for-byte (integer add is associative).
        let native: Vec<u8> = (0..NATIVE_N)
            .map(|i| ((i * 37 + (i / NATIVE_W) * 19 + (i % NATIVE_W) * 7) & 0xff) as u8)
            .collect();
        for size in [1, 7, 32, 84, 96, 128, 240] {
            assert_eq!(
                resize_area_gray(&native, NATIVE_W, NATIVE_H, size, size),
                resize_area_gray_int_reference(&native, NATIVE_W, NATIVE_H, size, size),
                "separable must match integer INTER_AREA oracle at {size}x{size}"
            );
        }
        let src: Vec<u8> = (0..17 * 11)
            .map(|i| ((i * 23 + (i / 17) * 11) & 0xff) as u8)
            .collect();
        assert_eq!(
            resize_area_gray(&src, 17, 11, 9, 7),
            resize_area_gray_int_reference(&src, 17, 11, 9, 7)
        );
    }

    #[test]
    fn integer_inter_area_within_one_of_legacy_f64_path() {
        // The integer path differs from the legacy f64 reference by at most 1 (exact-half ties only).
        let native: Vec<u8> = (0..NATIVE_N)
            .map(|i| ((i * 37 + (i / NATIVE_W) * 19 + (i % NATIVE_W) * 7) & 0xff) as u8)
            .collect();
        for size in [1, 7, 32, 84, 96, 128, 240] {
            let mine = resize_area_gray(&native, NATIVE_W, NATIVE_H, size, size);
            let f64ref = resize_area_gray_scalar_reference(&native, NATIVE_W, NATIVE_H, size, size);
            assert!(
                max_abs_diff(&mine, &f64ref) <= 1,
                "integer INTER_AREA must stay within 1 of the f64 reference at {size}x{size}"
            );
        }
    }

    #[test]
    fn identity_size_returns_source_unchanged() {
        let src: Vec<u8> = (0..12).collect();
        assert_eq!(resize_area_gray(&src, 4, 3, 4, 3), src);
    }

    #[test]
    fn gray_shape_accepts_rectangular_output() {
        let cfg = ObsConfig::gray_shape(4, 96, 72, true, RenderPolicy::TrainingSparse, false);
        assert_eq!(
            cfg.shape(),
            ObsShape {
                stack: 1,
                width: 96,
                height: 72,
                channels: 1
            }
        );
        let window = ObsWindow::new(cfg);
        assert_eq!(
            window.shape(),
            ObsShape {
                stack: 1,
                width: 96,
                height: 72,
                channels: 1
            }
        );
    }

    #[test]
    fn two_by_two_to_one_averages_all_four() {
        // (0 + 10 + 20 + 30) / 4 = 15
        assert_eq!(resize_area_gray(&[0, 10, 20, 30], 2, 2, 1, 1), vec![15]);
    }

    #[test]
    fn four_by_four_to_two_by_two_block_averages() {
        // Each 2x2 block is constant, so each destination pixel == that block.
        let src = vec![
            10, 10, 20, 20, //
            10, 10, 20, 20, //
            30, 30, 40, 40, //
            30, 30, 40, 40, //
        ];
        assert_eq!(resize_area_gray(&src, 4, 4, 2, 2), vec![10, 20, 30, 40]);
    }

    #[test]
    fn non_integer_ratio_uses_fractional_area_weights() {
        // 3->2 (sx=1.5): src=[0,60,120] -> [30/1.5, 150/1.5] = [20,100].
        assert_eq!(resize_area_gray(&[0, 60, 120], 3, 1, 2, 1), vec![20, 100]);
    }

    #[test]
    fn obs_window_accumulates_reward_and_maxpools_boundary_frames() {
        let mut window = ObsWindow::new(ObsConfig::gray(
            2,
            8,
            true,
            RenderPolicy::HumanVisible,
            false,
        ));
        let lives = [3, 0, 0, 0];
        window.reset(lives);
        let reset_frame = vec![1u8; NATIVE_N];
        window
            .refresh(FrameSample {
                rgb: None,
                gray: Some(&reset_frame),
                ram: None,
                rewards: [0.0; 4],
                lives,
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();

        let f1 = vec![10u8; NATIVE_N];
        let mut f2 = vec![20u8; NATIVE_N];
        f2[0] = 200;
        let s1 = window
            .push_frame(
                FrameSample {
                    rgb: None,
                    gray: Some(&f1),
                    ram: None,
                    rewards: [0.25, 0.0, 0.0, 0.0],
                    lives,
                    frame_number: 1,
                    episode_frame_number: 1,
                    terminated: false,
                    truncated: false,
                },
                false,
            )
            .unwrap();
        assert!(!s1.obs_step);
        let s2 = window
            .push_frame(
                FrameSample {
                    rgb: None,
                    gray: Some(&f2),
                    ram: None,
                    rewards: [0.75, 0.0, 0.0, 0.0],
                    lives,
                    frame_number: 2,
                    episode_frame_number: 2,
                    terminated: false,
                    truncated: false,
                },
                false,
            )
            .unwrap();
        assert!(s2.obs_step);
        assert_eq!(s2.rewards[0], 1.0);
        assert_eq!(window.observation(), compute_obs(&f2, Some(&f1), 8));
    }

    #[test]
    fn stack_pads_on_reset_and_rolls_newest_last() {
        // ale-py frame stacking: reset pads every slot, then each step rolls the newest frame in last.
        let cfg = ObsConfig::gray(1, 8, false, RenderPolicy::HumanVisible, false).with_stack_num(3);
        let frame_len = 8 * 8;
        let mut window = ObsWindow::new(cfg);
        let lives = [0, 0, 0, 0];
        window.reset(lives);

        fn feed(window: &mut ObsWindow, value: u8, n: u64) {
            let frame = vec![value; NATIVE_N];
            window
                .push_frame(
                    FrameSample {
                        rgb: None,
                        gray: Some(&frame),
                        ram: None,
                        rewards: [0.0; 4],
                        lives: [0, 0, 0, 0],
                        frame_number: n,
                        episode_frame_number: n,
                        terminated: false,
                        truncated: false,
                    },
                    false,
                )
                .unwrap();
        }

        // Reset frame == 1 -> stack padded to [1, 1, 1].
        let reset_frame = vec![1u8; NATIVE_N];
        window
            .refresh(FrameSample {
                rgb: None,
                gray: Some(&reset_frame),
                ram: None,
                rewards: [0.0; 4],
                lives,
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();
        assert_eq!(window.observation().len(), 3 * frame_len);
        assert!(window.observation().iter().all(|&b| b == 1));

        // Step frame == 2 -> [1, 1, 2].
        feed(&mut window, 2, 1);
        {
            let obs = window.observation();
            assert!(obs[..frame_len].iter().all(|&b| b == 1));
            assert!(obs[frame_len..2 * frame_len].iter().all(|&b| b == 1));
            assert!(obs[2 * frame_len..].iter().all(|&b| b == 2));
        }

        // Step frame == 3 -> [1, 2, 3].
        feed(&mut window, 3, 2);
        {
            let obs = window.observation();
            assert!(obs[..frame_len].iter().all(|&b| b == 1));
            assert!(obs[frame_len..2 * frame_len].iter().all(|&b| b == 2));
            assert!(obs[2 * frame_len..].iter().all(|&b| b == 3));
        }
    }

    #[test]
    fn stack_num_one_serves_single_frame() {
        // stack_num == 1 serves one frame, byte-identical to the pre-stacking pipeline.
        let cfg = ObsConfig::gray(1, 8, false, RenderPolicy::HumanVisible, false);
        let mut window = ObsWindow::new(cfg);
        window.reset([0, 0, 0, 0]);
        let frame = vec![7u8; NATIVE_N];
        window
            .refresh(FrameSample {
                rgb: None,
                gray: Some(&frame),
                ram: None,
                rewards: [0.0; 4],
                lives: [0, 0, 0, 0],
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();
        assert_eq!(window.observation().len(), 8 * 8);
        assert!(window.observation().iter().all(|&b| b == 7));
    }

    #[test]
    fn obs_window_can_end_on_life_loss() {
        let mut window = ObsWindow::new(ObsConfig::gray(
            4,
            8,
            false,
            RenderPolicy::HumanVisible,
            true,
        ));
        let frame = vec![1u8; NATIVE_N];
        window.reset([3, 0, 0, 0]);
        window
            .refresh(FrameSample {
                rgb: None,
                gray: Some(&frame),
                ram: None,
                rewards: [0.0; 4],
                lives: [3, 0, 0, 0],
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();

        let step = window
            .push_frame(
                FrameSample {
                    rgb: None,
                    gray: Some(&frame),
                    ram: None,
                    rewards: [0.0; 4],
                    lives: [2, 0, 0, 0],
                    frame_number: 1,
                    episode_frame_number: 1,
                    terminated: false,
                    truncated: false,
                },
                false,
            )
            .unwrap();
        assert!(step.terminated);
        assert!(step.obs_step);
    }

    #[test]
    fn life_loss_ignores_unused_player_slots() {
        // A 2-player window must not treat a drop in an unused trailing slot as a life loss (Serve false-reset bug).
        let mut cfg = ObsConfig::gray(4, 8, false, RenderPolicy::HumanVisible, true);
        cfg.players = 2;
        let mut window = ObsWindow::new(cfg);
        let frame = vec![1u8; NATIVE_N];
        window.reset([1, 1, 1, 4]);
        window
            .refresh(FrameSample {
                rgb: None,
                gray: Some(&frame),
                ram: None,
                rewards: [0.0; 4],
                lives: [1, 1, 1, 4],
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();
        // Unused slot 3 drops 4->3 (P1 moved left): must NOT terminate.
        let step = window
            .push_frame(
                FrameSample {
                    rgb: None,
                    gray: Some(&frame),
                    ram: None,
                    rewards: [0.0; 4],
                    lives: [1, 1, 1, 3],
                    frame_number: 1,
                    episode_frame_number: 1,
                    terminated: false,
                    truncated: false,
                },
                false,
            )
            .unwrap();
        assert!(
            !step.terminated,
            "a drop in an unused slot must not end the episode"
        );
        // A real active-port life loss (P1 alive 1->0) still terminates.
        let step = window
            .push_frame(
                FrameSample {
                    rgb: None,
                    gray: Some(&frame),
                    ram: None,
                    rewards: [0.0; 4],
                    lives: [0, 1, 1, 3],
                    frame_number: 2,
                    episode_frame_number: 2,
                    terminated: false,
                    truncated: false,
                },
                false,
            )
            .unwrap();
        assert!(
            step.terminated,
            "an active port's life loss must end the episode"
        );
    }

    #[test]
    fn obs_window_rgb_uses_native_shape() {
        let cfg = ObsConfig {
            obs_kind: ObsKind::RgbNative,
            render_policy: RenderPolicy::HumanVisible,
            ..ObsConfig::default()
        };
        let mut window = ObsWindow::new(cfg);
        let rgb: Vec<u8> = (0..NATIVE_N * 3).map(|i| (i & 0xff) as u8).collect();
        window
            .refresh(FrameSample {
                rgb: Some(&rgb),
                gray: None,
                ram: None,
                rewards: [0.0; 4],
                lives: [0; 4],
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();
        assert_eq!(
            window.shape(),
            ObsShape {
                stack: 1,
                width: NATIVE_W,
                height: NATIVE_H,
                channels: 3
            }
        );
        assert_eq!(window.observation(), rgb);
    }

    #[test]
    fn obs_window_rgb_can_resize() {
        let cfg = ObsConfig {
            obs_kind: ObsKind::rgb_shape(2, 1),
            render_policy: RenderPolicy::HumanVisible,
            ..ObsConfig::default()
        };
        let mut window = ObsWindow::new(cfg);
        let mut rgb = vec![0u8; NATIVE_N * 3];
        for y in 0..NATIVE_H {
            for x in 0..NATIVE_W {
                let offset = (y * NATIVE_W + x) * 3;
                rgb[offset] = x as u8;
                rgb[offset + 1] = y as u8;
                rgb[offset + 2] = 10;
            }
        }
        window
            .refresh(FrameSample {
                rgb: Some(&rgb),
                gray: None,
                ram: None,
                rewards: [0.0; 4],
                lives: [0; 4],
                frame_number: 0,
                episode_frame_number: 0,
                terminated: false,
                truncated: false,
            })
            .unwrap();
        assert_eq!(
            window.shape(),
            ObsShape {
                stack: 1,
                width: 2,
                height: 1,
                channels: 3
            }
        );
        assert_eq!(window.observation().len(), 6);
        assert_eq!(window.observation()[2], 10);
        assert_eq!(window.observation()[5], 10);
    }

    #[test]
    fn training_sparse_renders_only_observed_pixel_frames() {
        let cfg = ObsConfig::gray(4, 84, true, RenderPolicy::TrainingSparse, false);
        assert!(!cfg.should_render_subframe(0, 4));
        assert!(!cfg.should_render_subframe(1, 4));
        assert!(cfg.should_render_subframe(2, 4));
        assert!(cfg.should_render_subframe(3, 4));

        let no_pool = ObsConfig::gray(4, 84, false, RenderPolicy::TrainingSparse, false);
        assert!(!no_pool.should_render_subframe(2, 4));
        assert!(no_pool.should_render_subframe(3, 4));

        let human = ObsConfig::gray(4, 84, true, RenderPolicy::HumanVisible, false);
        assert!((0..4).all(|t| human.should_render_subframe(t, 4)));
    }
}
