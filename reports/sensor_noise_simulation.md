# Sensor Noise Simulation (Poisson + Read Noise)

2026-11-21

## Model

We simulate camera sensor noise as photon (Poisson) noise plus a small electronic (Gaussian read) noise, applied to each pixel of a clean frame x in [0, 1]:

```math
y = \frac{\mathrm{Poisson}(x \cdot P)}{P} + \mathcal{N}(0,\, r^2)
```

- **P (`photons`)** is the mean number of photons a pure white pixel collects. A pixel of brightness x collects on average x × P photons; the actual count is random (Poisson). Dividing by P maps the count back to brightness, so the noisy image has the same mean as the clean one.
- **r (`read`)** is the standard deviation of the read noise, in brightness units. We tie it to P as r = 0.6 / P, i.e. a read noise of 0.6 electrons, which is small next to the photon noise.

Photon noise has standard deviation √(x × P) / P = √(x / P) in brightness. Its size relative to the signal is 1 / √(x × P), so fewer photons give a grainier image, and bright pixels get more absolute noise but less relative noise than dark ones.

| P (photons) | Noise std on a mid-grey pixel (x = 0.5) |
| --- | --- |
| 17 | 0.17 (≈ 44 of 255 grey levels) |
| 130 | 0.062 (≈ 16 grey levels) |
| 1,900 | 0.016 (≈ 4 grey levels) |
| 5,000 | 0.010 (≈ 2.5 grey levels) |

## How a frame is corrupted

The noise is generated on the fly in the data loader (`third_party/BKinD/dataloader/noise.py`); no noisy frames are stored. For each frame:

1. **Resize first.** The frame is resized to the network input size (CalMS21: 256 × 256), then corrupted. Noise levels are therefore defined at input resolution.
2. **Scale to [0, 1].** The 8-bit image is divided by 255. Grayscale frames stored as RGB (all three channels equal) are reduced to one channel.
3. **Photon noise.** For every pixel, draw a Poisson count with mean x × P and divide it by P.
4. **Read noise.** Add an independent Gaussian value with mean 0 and standard deviation r to every pixel.
5. **Back to 8 bits.** Multiply by 255, round, clip to [0, 255] and store as uint8, as if the noisy frame were saved as PNG.
6. **Same noise in all channels.** For grayscale content the one noisy channel is copied to R, G and B.

In code, steps 3 and 4 are one line:

```python
y = rng.poisson(x * photons) / photons + rng.normal(0.0, read, x.shape)
```

Only the frames are corrupted, never the keypoint labels.

## Random seeds and frozen noise

Every frame's noise is deterministic: the random generator is seeded from the frame's identity, so a run can be repeated exactly.

- **Per-frame noise (default).** Seed = (noise seed, CRC32 of the video name, frame number). A frame gets the same noise in every epoch and every run, like a recorded noisy video. Different frames, including frames A and B of a training pair, get independent noise.
- **Frozen noise (`frozen: true`, the control).** Seed = (noise seed, CRC32 of the video name), without the frame number. Every frame of a video then gets the same noise pattern.

Frozen noise needs one extra step. numpy's `poisson()` uses a variable number of random draws per pixel depending on its brightness, so two frames that differ anywhere fall out of step and get different noise from there on. In frozen mode we instead draw each Poisson value from exactly one uniform number u per pixel, by inverse CDF:

```math
k = \min\{\, k \ge 0 : P(X \le k) \ge u \,\}, \quad X \sim \mathrm{Poisson}(x \cdot P)
```

The search starts at the normal approximation k₀ = ⌊x·P + √(x·P) · Φ⁻¹(u)⌋ and steps up or down to the exact value (same result as scipy's `poisson.ppf`). Unchanged pixels in two frames then get identical noise (correlation 1.000). The Gaussian read noise does not depend on the image, so it stays in step without changes.

## Noise levels

Noise strength is set by P alone (with r = 0.6 / P), chosen so the noisy frames reach a target mean SSIM against the clean frames. SSIM and PSNR are measured by `scripts/calibrate_noise.py` on 200 CalMS21 training frames (20 per video) at 256 × 256, after the uint8 rounding, with a Gaussian SSIM window (11 px, σ = 1.5) on grayscale.

| Setting | P (photons) | r (read) | Mean SSIM | PSNR (dB) |
| --- | --- | --- | --- | --- |
| SSIM 0.999 | 120,000 | 0.000005 | 0.999 | 54.1 |
| SSIM 0.99 | 10,000 | 0.00006 | 0.990 | 44.2 |
| SSIM 0.98 | 5,000 | 0.00012 | 0.980 | 41.3 |
| SSIM 0.95 | 1,900 | 0.00032 | 0.950 | 37.1 |
| SSIM 0.9 | 880 | 0.00068 | 0.901 | 33.8 |
| Level 1 | 370 | 0.00162 | 0.801 | 30.0 |
| Level 2 | 130 | 0.00462 | 0.607 | 25.5 |
| Level 3 | 51 | 0.0118 | 0.402 | 21.4 |
| Level 4 | 17 | 0.0353 | 0.199 | 16.6 |

Levels 1–4 are the calibrated levels in `NOISE_LEVELS` (`noise: {type: poisson_gaussian, level: N}`). The milder settings are given as explicit values (`noise: {type: poisson_gaussian, photons: P, read: r}`); their SSIM and PSNR were measured the same way on the same 200 frames (2026-11-21).
