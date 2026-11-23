"""Short constructors for CalMS21 training runs (used by RunPod runs files; the Kaggle and Colab
notebooks define the same helpers inline). Each returns a run dict for kaggle_bkind.run():
dict(name=..., seed=..., overrides=dict(noise=...)). Noise types and calibrated levels are in
third_party/BKinD/dataloader/noise.py.
"""


KINDS = dict(pg='poisson_gaussian', ll='low_light', sp='speckle', blur='motion_blur', h264='h264')
BENCH_SSIM = [99, 98, 95, 90]  # benchmark levels 1-4 (decided 2026-11-23); see noise.py BENCH_LEVELS


def bench(kind, ssim, seed, frozen=False):
    """A benchmark cell: noise type `kind` (pg, sp, blur, h264) at SSIM 99/98/95/90, with the exact
    parameters from noise.py BENCH_LEVELS written into the config. Named CalMS21_<kind>_ssim<NN>_s<seed>
    (the same names the earlier edge runs used, so they count as benchmark cells)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
    from dataloader.noise import BENCH_LEVELS
    params = BENCH_LEVELS[KINDS[kind]][BENCH_SSIM.index(ssim)]
    extra = dict(frozen=True) if frozen else {}
    return noisy(f'CalMS21_{kind}_ssim{ssim}' + ('_frozen' if frozen else ''), seed,
                 type=KINDS[kind], **params, **extra)


def warp(run):
    """The same run with our warping method (config CalMS21_warp.yaml) instead of BKinD:
    warp(bench('pg', 95, 1)) -> CalMS21_warp_pg_ssim95_s1, warp(clean(1)) -> CalMS21_warp_clean_s1."""
    return dict(run, name=run['name'].replace('CalMS21_', 'CalMS21_warp_', 1), base_config='CalMS21_warp.yaml')


def warp_motion(run):
    """Step 1b: warping with the reconstruction loss weighted by where things move
    (config CalMS21_warp_motion.yaml): warp_motion(clean(1)) -> CalMS21_warpm_clean_s1."""
    return dict(run, name=run['name'].replace('CalMS21_', 'CalMS21_warpm_', 1), base_config='CalMS21_warp_motion.yaml')


def warp_motion_sep(run, weight=100.0):
    """Step 1c: step 1b + BKinD's separation loss (pushes keypoints apart so unused ones spread out):
    warp_motion_sep(clean(1)) -> CalMS21_warpms_clean_s1. Weight 100: with 8 of 10 keypoints bunched
    together the term is ~6, ~10% of the ~60 total loss (in BKinD it is ~13% of its loss when collapsed)."""
    run = warp_motion(run)
    overrides = dict(run.get('overrides', {}), warp_separation_weight=weight)
    return dict(run, name=run['name'].replace('CalMS21_warpm_', 'CalMS21_warpms_', 1), overrides=overrides)


def clean(seed):
    return dict(name=f'CalMS21_clean_s{seed}', seed=seed)


def noisy(name, seed, **noise):
    """Any noise, e.g. noisy('CalMS21_ll_L2_frozen', 1, type='low_light', level=2, frozen=True)."""
    return dict(name=f'{name}_s{seed}', seed=seed, overrides=dict(noise=dict(noise, seed=noise.get('seed', 0))))


def level(kind, lvl, seed, frozen=False):
    """Calibrated level of a noise type. kind: pg, ll, sp, blur or h264."""
    types = dict(pg='poisson_gaussian', ll='low_light', sp='speckle', blur='motion_blur', h264='h264')
    extra = dict(frozen=True) if frozen else {}
    return noisy(f'CalMS21_{kind}_L{lvl}' + ('_frozen' if frozen else ''), seed,
                 type=types[kind], level=lvl, **extra)


def pg(lvl, seed):
    return level('pg', lvl, seed)


def pg_ssim(ssim, photons, seed):
    """Sensor noise with explicit photons (read noise = 0.6 electrons)."""
    return noisy(f'CalMS21_pg_ssim{ssim}', seed, type='poisson_gaussian', photons=photons, read=0.6 / photons)


def sp_ssim(ssim, looks, seed):
    return noisy(f'CalMS21_sp_ssim{ssim}', seed, type='speckle', looks=looks)


def blur_ssim(ssim, length, seed):
    """Motion blur of `length` pixels at 256 px (sub-pixel lengths allowed)."""
    return noisy(f'CalMS21_blur_ssim{ssim}', seed, type='motion_blur', length=length)
