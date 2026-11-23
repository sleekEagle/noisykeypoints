# Example runs file for scripts/runpod_bkind.py. Copy it to /workspace/runs.py on the pod and edit.
# One run per GPU: a pod with N GPUs trains up to N runs at once. Helpers (scripts/run_specs.py):
#   clean(seed), pg(level, seed), level(kind, level, seed, frozen=False),
#   pg_ssim(ssim, photons, seed), sp_ssim(ssim, looks, seed), blur_ssim(ssim, length, seed),
#   noisy(name, seed, type=..., ...)        any other noise settings
# A run whose folder already exists in /workspace/runs resumes from its checkpoint.

RUNS = [blur_ssim(90, 3.18, 1),     # SSIM ~0.90
        level('blur', 1, 1)]        # calibrated level 1: 5 px, SSIM 0.8
