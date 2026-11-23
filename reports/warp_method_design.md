# Method step 1: keypoint-driven warping + equivariance (design)

2026-11-23. Status: **implemented** (commit `a5dc6ba`), checked on CPU and in a short GPU test; first full runs queued on RunPod (see "Implementation status" at the end). Research plan, Method items 2 and 6 ("build order: 2 and 6 first on top of the existing BKinD code").

## Goal

Replace BKinD's decoder, which **paints** an image, with one that may only **move pixels** of frame A to produce frame B. Keypoints then have to explain motion. Fresh per-frame noise can't be explained by moving pixels, so it should not pull keypoints off the animals.

This step deliberately changes nothing else:
- the same CalMS21 frame pairs (A = frame i, B = frame i+6, a pair every 7th frame);
- the same 256×256 input, 10 keypoints and noise code;
- the same keypoint encoder.

That way any difference from BKinD comes from the warping. Clip encoders, the confidence mask, temporal losses and feature-space losses come later, one at a time.

## What BKinD does now (for contrast)

1. A ResNet-50 encoder + `globalNet` turn each frame into K = 10 heatmaps (64×64). A keypoint is the heatmap-weighted mean position ("soft argmax").
2. The decoder gets frame A's ResNet features plus Gaussian blobs at the keypoints of A and B. It **generates** the SSIM map between A and B (BKinD's "what changed" target), trained with MSE + VGG perceptual loss.
3. From epoch 4, the rotated copies of B (90°, 180°, −90°) must give rotated heatmaps (rotation equivariance), and a separation loss keeps keypoints apart.

With independent noise in A and B, step 2's target is noise everywhere. The generator can lower its loss by painting noise-like texture, so keypoints aren't needed to explain the real motion. Our CalMS21 runs show the collapse, while frozen noise (identical in A and B) trains fine.

## Background: keypoints, Jacobians and warping, in plain terms

This method follows the First Order Motion Model (FOMM; Siarohin et al., NeurIPS 2019).

### What the Jacobian is
A keypoint's position says *where* a body part is. When a mouse turns its head, the small patch of fur around the head keypoint also **rotates**; when it stretches or squashes, the patch **stretches**. A shift alone can't describe that. Each keypoint therefore also carries a 2×2 matrix J, its **Jacobian**, describing how a small neighbourhood around it is turned and stretched. A point a small offset d from the keypoint moves to J·d:

| J | What happens to the patch |
| --- | --- |
| [[1, 0], [0, 1]] | nothing (identity) |
| [[cos θ, −sin θ], [sin θ, cos θ]] | rotated by θ |
| [[1.2, 0], [0, 1]] | stretched 20% sideways |

Position (2 numbers) plus Jacobian (4 numbers) is a **first-order** description of the motion near the keypoint, hence the name "first order motion".

### How the network estimates it
Nothing measures the Jacobian directly, and there are no labels: the network **learns to predict** it, and is trained only by whether it helps warp one frame into the next.
- The keypoint network already outputs one heatmap per keypoint (bright where it thinks the keypoint is); the position is the heatmap-weighted mean of pixel positions (soft argmax).
- For the Jacobian it outputs **4 extra maps per keypoint**, one per matrix entry. Each entry is read off as the average of its map **weighted by the keypoint's heatmap**:

  J_k = Σ_pixels z heatmap_k(z) · Jmaps_k(z)

  In effect the network says "this is how the patch is turned and stretched" at every pixel, and the answer is read out where the keypoint is.

### How a frame is warped with it
To make frame B from frame A, every pixel z of B asks where in A it comes from. Near keypoint k the answer is

  z_A ≈ p_k^A + J_k^A (J_k^B)⁻¹ (z − p_k^B):

take z's offset from the keypoint in B, undo B's local turn/stretch, apply A's, and add the result to the keypoint's position in A. Each keypoint gives such a local rule, and a background rule says "don't move". A small network (the dense motion network) decides which rule owns each pixel and blends them into one smooth motion field; frame A is then resampled along that field.

### How it learns without labels
1. **Reconstruction.** The warped frame A must look like the real frame B. A wrong Jacobian puts pixels in the wrong place, the error rises, and gradient descent pushes J towards the real local motion.
2. **Equivariance.** Bend a frame with a known random warp T (rotation, stretch, slight bending) and run the network on both versions. The keypoints must move by T, and the Jacobians must change by T's own local stretch, J(T(image)) ≈ ∇T · J(image). ∇T is known exactly because we chose T. This stops the network from inventing arbitrary matrices.

### Why this should ignore noise
FOMM also has a generator that **paints in** regions that moving pixels can't explain (things coming into view). Our decoder may not paint. Fresh per-frame noise can't be produced by moving the pixels of frame A, because B's noise is new and random, so the only way to lower the error is to get the **real** motion right. BKinD's decoder paints its target and is tempted to spend keypoints on noise.

The cost: newly visible areas (bedding uncovered by a moving mouse) can't be produced by moving pixels either, so some error remains there. The plan's confidence mask (method step 2) is meant to let the model mark such pixels as "can't be explained by motion" instead of distorting the keypoints to fit them.

## The new model (FOMM motion module, no generator)

```
frame A ─┐                                  ┌─> local affine maps T_k (shift + 2×2 Jacobian)
         ├─ keypoint net (BKinD's) ─> kp, J ┤
frame B ─┘                                  └─> dense motion net ─> flow field + blend masks
                                                                        │
                        predicted B = grid_sample(frame A, flow) <──────┘   (moves pixels only)
loss = perceptual(predicted B, B) + equivariance
```

### 1. Keypoints and Jacobians
- **Positions:** unchanged from BKinD (same encoder, `globalNet`, softmax heatmaps, soft argmax). So keypoints come out the same way as in the BKinD runs, and the same scoring code works.
- **Jacobians (new):** a 3×3 conv on `globalNet`'s last feature map (256 channels, 64×64) outputs 4 maps per keypoint. Each keypoint's 2×2 Jacobian is the heatmap-weighted average of its 4 maps, as in FOMM. The conv starts with zero weights and an identity bias, so every Jacobian starts as "no rotation, no stretch".

### 2. Local motions
For each keypoint k, the pixel z of B is fetched from A at

  T_k(z) = p_k^A + J_k^A (J_k^B)⁻¹ (z − p_k^B).

A (K+1)-th "background" motion is the identity, T_0(z) = z. CalMS21 has a fixed camera, so most of the image doesn't move.

### 3. Dense motion network (FOMM's)
- **Input, at 64×64:**
  - for each keypoint, the difference of its Gaussian blob in B and in A (K channels, + 1 zero channel for the background);
  - frame A warped by each of the K+1 local motions ((K+1)×3 channels).
- **Network:** a small hourglass (U-Net); FOMM uses 64 base channels, max 1024, 5 blocks.
- **Output:** K+1 masks (softmax per pixel) that say which motion "owns" each pixel. The flow is the mask-weighted sum of the K+1 local motions, upsampled to 256×256.

### 4. Move-only decoder
predicted B = `grid_sample(frame A, flow)`, at 256×256, bilinear. No generator, no occlusion map and no in-painting: the plan forbids painting pixels. Regions that become visible (bedding uncovered by a moving mouse) stay unexplained. That is the job of step 3 of the plan, the confidence mask.

### 5. Losses
- **Reconstruction:** FOMM's multi-scale perceptual loss. VGG-19 features of predicted B vs B at 256, 128, 64 and 32 px, L1, weight 10 per scale. BKinD's VGG wrapper (`model/vggforLoss.py`) is reused.
- **Equivariance (replaces BKinD's rotation passes and curriculum):** apply a random smooth warp T to B on the GPU (FOMM's: random affine with σ 0.05 plus thin-plate-spline bending with σ 0.005 on a 5×5 grid).
  - Positions must follow: kp(T(B)) ≈ T(kp(B)). L1, weight 10.
  - Jacobians must follow: J(T(B)) ≈ ∇T · J(B), with ∇T known exactly. L1 on (J(B))⁻¹ ∇T⁻¹ J(T(B)) vs the identity, weight 10.
  - This covers rotation and more, from epoch 0, with no curriculum.
- **Separation loss:** off at first (FOMM doesn't use one). Turn it on only if keypoints bunch up.

### 6. Training
- **Data:** same pairs, noise and batch size (5) as BKinD; 15 epochs.
- **Optimizer:** **Adam, lr 2e-4** (FOMM's). BKinD's SGD at lr 1e-3 was tuned for its own decoder; the new hourglass trains from scratch.
- **Precision:** mixed precision for the networks, as in our BKinD runs. Warping, Jacobian inverses and all losses stay float32. Inverting a 2×2 matrix in float16 is fragile.
- **Cost:** 3 encoder passes per step (A, B, T(B)), versus BKinD's 2 + 3 rotations, plus the small hourglass. Expect a run time similar to BKinD's: about 2 h on the RTX 4090.

## Code layout (BKinD stays as it is)

| File | Contents |
| --- | --- |
| `model/warp_model.py` | `WarpModel`: BKinD encoder + `globalNet` + Jacobian head, local motions, dense motion net, `grid_sample`. Same `get_keypoints()` as BKinD's model, so scoring works unchanged. |
| `model/dense_motion.py` | Hourglass and dense motion network (adapted from FOMM, MIT licence; credited in the file). |
| `loss/warp_loss.py` | Perceptual pyramid, random TPS, equivariance losses. |
| `train_video.py` | `method: warp` in the config picks the new model, loss and Adam; default `method: bkind` changes nothing. |
| `config/CalMS21_warp.yaml` | `CalMS21_clean.yaml` + `method: warp` + Adam settings. |
| `scripts/run_specs.py` | `warp(...)` run helper; run names `CalMS21_warp_<condition>_s<seed>`. |

## How it is tested

1. **On the pod, between queued BKinD runs:** a 1-epoch smoke test (shapes, memory, time per step, loss falls). Then a check that the flow is the identity at the start (predicted B ≈ A).
2. **First comparison, seed 1:** clean, plus sensor noise at SSIM 0.98 (BKinD: partly works) and 0.95 (BKinD: collapsed). The step passes if keypoints stay on the mice where BKinD collapsed, judged by eye and by the keypoint score once the scoring script exists.
3. The plan's gate for each step is the **echo validation set**. Echo (CAMUS / HMC-QU) isn't set up yet, so step 1 is judged on CalMS21 for now.

## Risks and what to watch

- **Mice are small.** Most of the 256×256 frame is still bedding that the background (identity) motion already explains. The mice give a small share of the loss, so keypoints may learn slowly or settle on the background. FOMM's usual videos (faces, bodies) fill the frame. If it happens: weight the loss toward pixels that changed between A and B, but only by a coarse, noise-insensitive measure, or the noise sensitivity comes back.
- **Blurring as a shortcut under noise.** Bilinear sampling at fractional positions averages neighbouring noisy pixels, which slightly lowers the error against B's independent noise. The model could prefer smeared, sub-pixel flows. The VGG loss and the coarse scales dampen this. Check the flow on noisy runs for a pattern of tiny shifts everywhere.
- **Uncovered regions:** unavoidable error where a mouse moves away. Acceptable in step 1; the confidence mask (step 2 of the build order) handles it.
- **Jacobian blow-ups:** a near-singular J_B makes J_B⁻¹ explode. Clamp the determinant and compute in float32.

## Implementation status (2026-11-23)

### Code (commit `a5dc6ba`; all paths under `third_party/BKinD/`)
| File | What it holds |
| --- | --- |
| `model/warp_model.py` | `WarpModel`. `detect()` returns keypoints (`value` (N, K, 2) as (x, y) in [−1, 1], `jacobian` (N, K, 2, 2), `heatmap`, `confidence`). `forward(A, B, transformed)` returns the prediction (A resampled along the flow, `grid_sample`, `padding_mode='border'`), the keypoints of A, B and T(B), the flow and the blend masks. The Jacobian head is a 3×3 conv on `globalNet`'s last 256-channel feature map (64×64), with zero weights and an identity bias. |
| `model/dense_motion.py` | From FOMM: `Hourglass` (64 base channels, max 1024, 5 blocks), `AntiAliasInterpolation2d`, `kp2gaussian`, `make_coordinate_grid`, `DenseMotionNetwork` (K+1 motions, motion 0 = identity, softmax masks, flow at 64×64). Also `safe_inverse_2x2`, which keeps \|det\| ≥ 1e-3. No occlusion map. `align_corners=True` everywhere, to match BKinD's `linspace(−1, 1)` coordinates. |
| `loss/warp_loss.py` | From FOMM: `Transform` (random affine σ 0.05 + thin-plate spline σ 0.005 on 5×5 points; `transform_frame`, `warp_coordinates`, `jacobian` by autograd). `WarpLoss`: VGG-19 perceptual loss (BKinD's `VGG_feat`, 5 layers) at 4 scales (256/128/64/32 px), L1, weight 10. Keypoint equivariance: L1 of kp(B) − T(kp(T(B))), weight 10. Jacobian equivariance: L1 of I − J(B)⁻¹ ∇T J(T(B)), weight 10. Optional separation, weight 0. |
| `train_video.py` | `method: warp` picks `WarpModel`, `WarpLoss` and Adam (lr from config, betas 0.5/0.999). Each step: random `Transform` of B, forward under autocast, loss in float32. It prints the loss parts every 500 steps and saves samples like BKinD (`recon_*` = the warped prediction). `method: bkind` (default) is unchanged. |
| `config/CalMS21_warp.yaml` | `CalMS21_clean.yaml` + `method: warp`, lr 2e-4, the `warp_*` loss weights and warp settings. |
| `scripts/run_specs.py` | `warp(run)`: the same run with our method, e.g. `warp(bench('pg', 95, 1))` → `CalMS21_warp_pg_ssim95_s1`. It sets the per-run `base_config` that `kaggle_bkind.make_configs` reads. |

Mixed precision: the ResNet, `globalNet`, Jacobian head, hourglass and VGG run in float16. Keypoint maths, Jacobians and their inverses, sparse motions, `grid_sample`, the random transform and all losses run in float32.

### Checks so far
- **CPU maths checks (laptop, no training):**
  - identical keypoints in A and B → every motion is the identity;
  - shifted keypoints → motions shift by exactly that amount;
  - `safe_inverse_2x2` = `torch.inverse`;
  - initial Jacobians = identity;
  - the same frame in and out → prediction = source (max difference 2e-4);
  - the transform's Jacobian = its affine part when there is no bending.
- **GPU test on the RunPod RTX 4090** (1,570 steps, sharing the GPU with a BKinD run):
  - runs without errors under mixed precision, ~0.06 s/step, 3.7 GB;
  - the loss falls from 93 to ~26, and the equivariance terms fall (positions 1.28 → 0.35, Jacobians 0.69 → 0.49).
  - Early samples: the prediction already looks like B, because the identity background explains most pixels. The keypoints still sit mid-arena and on the metal cylinder (the "mice are small" risk above); it is too early to judge.

### First full runs (queued on RunPod, seed 1)
`CalMS21_warp_clean_s1`, `CalMS21_warp_pg_ssim98_s1` and `CalMS21_warp_pg_ssim95_s1`, ~1.5 h each on the 4090. They are compared with the BKinD runs at the same settings (clean works; 0.98 partly works; 0.95 collapsed).
