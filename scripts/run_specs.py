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


def warp_motion(run, tau=None, floor=None):
    """Step 1b: warping with the reconstruction loss weighted by where things move
    (config CalMS21_warp_motion.yaml): warp_motion(clean(1)) -> CalMS21_warpm_clean_s1.
    tau / floor change the motion weight's 'full motion' threshold (default 0.05) and the weight of
    still pixels (default 0.05); they go into the name, e.g. warp_motion(clean(1), tau=0.02, floor=0.01)
    -> CalMS21_warpm_t02f01_clean_s1."""
    tag, overrides = '', dict(run.get('overrides', {}))
    if tau is not None:
        tag += f't{round(tau * 100):02d}'
        overrides['warp_motion_tau'] = tau
    if floor is not None:
        tag += f'f{round(floor * 100):02d}'
        overrides['warp_motion_floor'] = floor
    prefix = 'CalMS21_warpm_' + (tag + '_' if tag else '')
    out = dict(run, name=run['name'].replace('CalMS21_', prefix, 1), base_config='CalMS21_warp_motion.yaml')
    if overrides:
        out['overrides'] = overrides
    return out


def warp_motion_sep(run, weight=100.0):
    """Step 1c: step 1b + BKinD's separation loss (pushes keypoints apart so unused ones spread out):
    warp_motion_sep(clean(1)) -> CalMS21_warpms_clean_s1. Weight 100: with 8 of 10 keypoints bunched
    together the term is ~6, ~10% of the ~60 total loss (in BKinD it is ~13% of its loss when collapsed)."""
    run = warp_motion(run)
    overrides = dict(run.get('overrides', {}), warp_separation_weight=weight)
    return dict(run, name=run['name'].replace('CalMS21_warpm_', 'CalMS21_warpms_', 1), overrides=overrides)


def warp_track(run, weight=None):
    """Step 1d: step 1b + the point-track loss (config CalMS21_warp_track.yaml; needs tracks precomputed
    for the run's noise by scripts/precompute_tracks.py): warp_track(bench('pg', 95, 1)) ->
    CalMS21_warpt_pg_ssim95_s1. weight overrides warp_track_weight (default 100) and goes into the name."""
    overrides = dict(run.get('overrides', {}))
    tag = ''
    if weight is not None:
        overrides['warp_track_weight'] = weight
        tag = f'w{weight:g}_'
    out = dict(run, name=run['name'].replace('CalMS21_', f'CalMS21_warpt_{tag}', 1),
               base_config='CalMS21_warp_track.yaml')
    if overrides:
        out['overrides'] = overrides
    return out


def track_target(run, sigma=4.0, peak=0.5, full=4.0, tracks_dir='/workspace/tracks'):
    """BKinD whose decoder redraws a map of blobs at the CoTracker3-tracked moving points (positions in A
    and B) instead of its 1 - SSIM map (loss/change_target.py; needs tracks precomputed for the run's
    noise): track_target(bench('pg', 95, 1)) -> CalMS21_tt_pg_ssim95_s1."""
    overrides = dict(run.get('overrides', {}), change_target=dict(type='tracks', sigma=sigma, peak=peak, full=full),
                     tracks=dict(dir=tracks_dir, max_points=512))
    return dict(run, name=run['name'].replace('CalMS21_', 'CalMS21_tt_', 1), overrides=overrides)


def coarse_target(run, factor=4):
    """Control without a tracker: BKinD's target computed on frames averaged down `factor`x
    (loss/change_target.py): coarse_target(bench('pg', 95, 1)) -> CalMS21_ct4_pg_ssim95_s1."""
    overrides = dict(run.get('overrides', {}), change_target=dict(type='coarse', factor=factor))
    return dict(run, name=run['name'].replace('CalMS21_', f'CalMS21_ct{factor}_', 1), overrides=overrides)


def multi_scale(run, factors=(1, 2, 4), sigma='global', normalize=False):
    """BKinD with one decoder head per change-target resolution (full, 2x, 4x averaged), each weighted by
    a learned uncertainty (model/scale_sigma.py). sigma 'global' = one number per head -> CalMS21_msg_*,
    'image' = predicted per image pair -> CalMS21_msi_*: multi_scale(bench('pg', 95, 1), sigma='image')."""
    tag = {'global': 'msg', 'image': 'msi'}[sigma] + ('n' if normalize else '')   # n: total weight fixed at 1
    ms = dict(factors=list(factors), sigma=sigma, **({'normalize': True} if normalize else {}))
    overrides = dict(run.get('overrides', {}), multi_scale=ms)
    return dict(run, name=run['name'].replace('CalMS21_', f'CalMS21_{tag}_', 1), overrides=overrides)


def feature_target(run, size=384, gain=1.0):
    """BKinD whose decoder redraws the change in CoTracker3's frozen encoder features between A and B
    (loss/change_target.py): feature_target(bench('pg', 95, 1)) -> CalMS21_ft_pg_ssim95_s1."""
    overrides = dict(run.get('overrides', {}), change_target=dict(type='feature', size=size, gain=gain))
    return dict(run, name=run['name'].replace('CalMS21_', 'CalMS21_ft_', 1), overrides=overrides)


def temporal(run, k=3, frames='/root/data/CalMS21_allframes/train'):
    """BKinD's change target built from the mean of k frames centred on A and on B (the network still sees
    single frames; dataloader/mouse_dataset.py). Combine with coarse_target for the shrunk version:
    temporal(coarse_target(bench('pg', 95, 1))) -> CalMS21_t3_ct4_pg_ssim95_s1. Needs every-frame data."""
    overrides = dict(run.get('overrides', {}), temporal=dict(k=k, frames=frames), workers=6)
    return dict(run, name=run['name'].replace('CalMS21_', f'CalMS21_t{k}_', 1), overrides=overrides)


def multi_frame(run, k=3, step=2, frames='/root/data/CalMS21_allframes/train'):
    """The keypoint network sees k frames: the current one and k-1 earlier ones, `step` frames apart
    (model/multiframe.py; needs every-frame data). Combine with coarse_target:
    multi_frame(coarse_target(bench('pg', 95, 1))) -> CalMS21_mf3_ct4_pg_ssim95_s1."""
    overrides = dict(run.get('overrides', {}), input_frames=dict(k=k, step=step, frames=frames), workers=6)
    return dict(run, name=run['name'].replace('CalMS21_', f'CalMS21_mf{k}_', 1), overrides=overrides)


def bg_mask(run, sigma=2.0, tau=0.1, step=60, window=600, seg=300):
    """Loss only where frame A or B differs from the local background (per-pixel median of nearby
    noisy frames; dataloader/background.py). Works for BKinD and warp runs:
    bg_mask(bench('pg', 95, 1)) -> CalMS21_bgm_pg_ssim95_s1 (BKinD),
    bg_mask(warp(clean(1))) -> CalMS21_bgm_warp_clean_s1 (replaces the two-frame weight; floor 0.05)."""
    overrides = dict(run.get('overrides', {}),
                     motion_mask=dict(sigma=sigma, tau=tau, step=step, window=window, seg=seg))
    return dict(run, name=run['name'].replace('CalMS21_', 'CalMS21_bgm_', 1), overrides=overrides)


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
