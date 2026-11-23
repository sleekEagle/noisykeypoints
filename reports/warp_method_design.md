# Method step 1: keypoint-driven warping + equivariance (design)

2026-11-23. Status: design, not yet implemented. Research plan, Method items 2 and 6 ("build order: 2 and 6 first on top of the existing BKinD code").

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
