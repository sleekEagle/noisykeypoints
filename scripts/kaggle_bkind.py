"""Train (and, for JHMDB, score) BKinD on Kaggle. The notebooks notebooks/bkind_*_kaggle.ipynb only
set the runs, clone this repo and call run(); everything else lives here, so changes reach Kaggle
with a git push and the notebooks stay short.

    import kaggle_bkind
    kaggle_bkind.run('calms21', [dict(name='CalMS21_clean_s1', seed=1)], t0=T0)

Each run is a dict with a unique `name` (its output folder under /kaggle/working/runs, also used to
resume) and a `seed`. Optional keys:
  overrides   settings that differ from the dataset's config, e.g. dict(nkpts=20)
  start_from  name of a finished run (attached as input) to continue training from
  tune        JHMDB only: score on held-out train clips instead of the test set
Runs train at the same time, one per GPU. If a previous version's output is attached as input, each
run resumes from its own checkpoint. Training stops cleanly before the session's time limit.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BK = os.path.join(REPO, 'third_party', 'BKinD')
INPUT = '/kaggle/input'

DATASETS = {
    # config: BKinD config with the experiment settings; reserve: hours kept free after training
    # (scoring and saving the output); poll: minutes between progress lines.
    'calms21': dict(config='CalMS21_clean.yaml', reserve=0.5, poll=5),
    'jhmdb': dict(config='JHMDB.yaml', reserve=0.7, poll=2),
}


def sh(cmd, **kw):
    """Run a command, showing its output in the notebook."""
    subprocess.run(cmd, check=True, **kw)


def find_dir(test, top=None):
    for root, dirs, files in os.walk(top or INPUT):
        if test(root, dirs, files):
            return root


def copy_tree(src, dst, last=None, workers=32):
    """Copy a folder with many small files, `workers` files at a time (/kaggle/input is slow to read
    file by file). The file named `last` is copied at the end, marking a complete copy."""
    t = time.time()
    jobs = []
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        jobs += [(os.path.join(root, f), os.path.join(dst, rel, f)) for f in files
                 if not (rel == '.' and f == last)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: shutil.copyfile(*j), jobs))
    if last:
        shutil.copyfile(os.path.join(src, last), os.path.join(dst, last))
    print(f'copied {len(jobs)} files in {(time.time() - t) / 60:.1f} min')


# ---------------------------------------------------------------- data

def prepare_calms21(data):
    """Copy the uploaded CalMS21 frames (made by prepare_calms21_bkind.py: a folder with videos.json
    and the mouse0xx_* folders, or a zip of it) to `data` and check them. Returns the train folder."""
    split = f'{data}/CalMS21_bkind/train'
    if not os.path.exists(f'{split}/videos.json'):  # missing = never copied, or copy cut off
        is_src = lambda r, ds, fs: 'videos.json' in fs and any(d.startswith('mouse0') for d in ds)
        src = find_dir(is_src)
        if src is None:
            z = glob.glob(f'{INPUT}/**/*.zip', recursive=True)
            assert z, f'could not find the CalMS21 frames (folder with videos.json, or a zip) under {INPUT}'
            zipfile.ZipFile(z[0]).extractall(f'{data}/unzipped')
            src = find_dir(is_src, f'{data}/unzipped')
        print('CalMS21 frames found in', src)
        shutil.rmtree(split, ignore_errors=True)
        copy_tree(src, split, last='videos.json')
    info = json.load(open(f'{split}/videos.json'))['videos']
    for v, r in info.items():
        n = len(glob.glob(f'{split}/{v}/*.png'))
        assert n == r['saved_frames'], f'{v}: {n} frames on disk, expected {r["saved_frames"]}'
    print('data ready:', len(info), 'videos,', sum(r['saved_frames'] for r in info.values()), 'frames')
    return split


def prepare_jhmdb(data):
    """Find the 3 JHMDB uploads (JHMDB_rgb, joint_positions, splits; folders or zips) and build
    split 1 with prepare_jhmdb_bkind.py. Returns (split folder, raw JHMDB folder)."""
    def find_or_unzip(name, test):
        d = find_dir(test)
        if d is None:
            z = glob.glob(f'{INPUT}/**/{name}.zip', recursive=True)
            assert z, f'could not find {name} (folder or zip) under {INPUT}'
            out = f'{data}/unzipped/{name}'
            zipfile.ZipFile(z[0]).extractall(out)
            d = find_dir(test, out)
        print(f'{name}: {d}')
        return d

    def has_clip_file(r, ds, pattern):
        return 'brush_hair' in ds and glob.glob(os.path.join(r, 'brush_hair', '*', pattern))

    frames = find_or_unzip('JHMDB_rgb', lambda r, ds, fs: has_clip_file(r, ds, 'image_00001.jpg'))
    joints = find_or_unzip('joint_positions', lambda r, ds, fs: has_clip_file(r, ds, 'joint_positions.mat'))
    splits = find_or_unzip('splits', lambda r, ds, fs: 'brush_hair_test_split1.txt' in fs)

    # The layout prepare_jhmdb_bkind.py expects, using symlinks (no copying).
    jhmdb = f'{data}/JHMDB'
    os.makedirs(jhmdb, exist_ok=True)
    for name, src in [('frames', frames), ('joint_positions', joints), ('splits', splits)]:
        if not os.path.exists(f'{jhmdb}/{name}'):
            os.symlink(src, f'{jhmdb}/{name}')

    # clips.json is written last, so a folder without it is a half-finished build: start again.
    split = f'{data}/JHMDB_bkind/split1'
    if os.path.exists(split) and not os.path.exists(f'{split}/clips.json'):
        print('removing an incomplete build')
        shutil.rmtree(split)
    if not os.path.exists(f'{split}/clips.json'):
        t = time.time()
        sh([sys.executable, f'{REPO}/scripts/prepare_jhmdb_bkind.py', '--jhmdb', jhmdb,
            '--out', f'{data}/JHMDB_bkind', '--split', '1'])
        print(f'took {(time.time() - t) / 60:.1f} min')
    assert os.path.exists(f'{split}/clips.json'), 'data preparation failed (see the output above)'
    print('data ready:', len(os.listdir(f'{split}/train')), 'train clips,',
          len(os.listdir(f'{split}/val')), 'val clips')
    return split, jhmdb


# ---------------------------------------------------------------- training

def make_configs(runs, config, split, out, train_hours):
    """Write one config per run: the experiment settings from `config`, plus the run's overrides,
    with this session's paths, seed and time limit."""
    import torch
    base = yaml.safe_load(open(f'{BK}/config/{config}'))
    n_gpu = torch.cuda.device_count()
    assert len(runs) <= n_gpu, f'{len(runs)} runs but only {n_gpu} GPU(s): one run per GPU'
    for i, r in enumerate(runs):
        r['dir'] = f'{out}/{r["name"]}'
        os.makedirs(r['dir'], exist_ok=True)
        cfg = dict(base, **r.get('overrides', {}))
        cfg.update(data=split, checkpoint=r['dir'], gpu=0, workers=2,  # 2 loader workers each: Kaggle has 4 CPUs
                   seed=r['seed'], max_hours=round(train_hours, 2))
        r['cfg'] = cfg
        r['config'] = f'{BK}/config/kaggle_{r["name"]}.yaml'
        yaml.safe_dump(cfg, open(r['config'], 'w'))
        print(f'GPU {i}: {r["name"]}  seed={r["seed"]}  batch={cfg["batch_size"]}  amp={cfg["amp"]}  '
              f'epochs={cfg["epochs"]}  time limit={cfg["max_hours"]} h  changes={r.get("overrides", {})}'
              + (f'  continues from {r["start_from"]}' if r.get('start_from') else '')
              + ('  [tuning: scored on held-out train clips]' if r.get('tune') else ''))


def train(runs, poll_minutes, t0):
    """Start every run on its own GPU (resuming where possible) and print progress until all end."""
    for i, r in enumerate(runs):
        ckpt = f'{r["dir"]}/checkpoint.pth.tar'
        if not os.path.exists(ckpt):
            # Resume this run if a previous session saved it; otherwise, if start_from is set,
            # start from that run's checkpoint (log.txt, samples and checkpoint are copied over).
            for src in [r['name']] + ([r['start_from']] if r.get('start_from') else []):
                prev = glob.glob(f'{INPUT}/**/runs/{src}/checkpoint.pth.tar', recursive=True)
                if prev:
                    shutil.copytree(os.path.dirname(prev[0]), r['dir'], dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('eval', 'model_best.pth.tar'))
                    print(f'{r["name"]}: copied checkpoint of {src} from {os.path.dirname(prev[0])}')
                    break
            else:
                assert not r.get('start_from'), \
                    f'{r["name"]}: start_from run {r["start_from"]} not found in {INPUT}'
        cmd = [sys.executable, 'train_video.py', '--config', r['config']]
        if os.path.exists(ckpt):
            cmd += ['--resume', ckpt]
        env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(i))  # sees only GPU i
        r['log'] = open(f'{r["dir"]}/train_log.txt', 'a')
        r['proc'] = subprocess.Popen(cmd, cwd=BK, env=env, stdout=r['log'], stderr=subprocess.STDOUT)
        print(f'GPU {i}: started {r["name"]}', '(resuming)' if '--resume' in cmd else '(fresh)')

    def last_line(path, keys=('Epoch:', 'epoch ', 'STOPPED', 'COMPLETE', 'Error')):
        lines = [l.rstrip() for l in open(path, errors='replace') if l.startswith(keys) or 'Error' in l]
        return lines[-1] if lines else '(starting)'

    while any(r['proc'].poll() is None for r in runs):
        time.sleep(poll_minutes * 60)
        print(time.strftime('%H:%M'), f'({(time.time() - t0) / 3600:.1f} h into session)', flush=True)
        for r in runs:
            print(f'   {r["name"]}: {last_line(r["dir"] + "/train_log.txt")}', flush=True)
    for r in runs:
        r['log'].close()
        print(f'{r["name"]}: exit code {r["proc"].returncode}')
        if r['proc'].returncode != 0:
            print(''.join(open(f'{r["dir"]}/train_log.txt', errors='replace').readlines()[-30:]))


def report(runs):
    """Print each run's training curve (log.txt) and how many epochs are done. Sets r['finished']."""
    import torch
    for r in runs:
        print('=====', r['name'])
        if os.path.exists(f'{r["dir"]}/log.txt'):
            print(open(f'{r["dir"]}/log.txt').read())
        ckpt = f'{r["dir"]}/checkpoint.pth.tar'
        epoch = torch.load(ckpt, map_location='cpu', weights_only=False)['epoch'] if os.path.exists(ckpt) else 0
        r['finished'] = epoch >= r['cfg']['epochs']
        print(f'epochs done: {epoch}/{r["cfg"]["epochs"]}',
              '' if r['finished'] else ' -> attach this output and run again to resume')


def score_jhmdb(runs, split, jhmdb):
    """Score each finished JHMDB run into <run>/eval (test clips, or held-out train clips if tune)."""
    for r in runs:
        if not r['finished']:
            continue
        print('=====', r['name'], '(scoring)')
        cmd = [sys.executable, 'scripts/eval_bkind_jhmdb.py', '--checkpoint', f'{r["dir"]}/checkpoint.pth.tar',
               '--data', split, '--jhmdb', jhmdb, '--out', f'{r["dir"]}/eval',
               '--nkpts', str(r['cfg']['nkpts']), '--image-size', str(r['cfg']['image_size'])]
        sh(cmd + (['--tune'] if r.get('tune') else []), cwd=REPO)  # tuning runs never see the test set
        vis = f'{r["dir"]}/eval/vis_val.png'
        if os.path.exists(vis):
            try:
                from IPython.display import Image, display
                display(Image(vis))
            except ImportError:
                pass


def run(dataset, runs, max_hours=11.5, t0=None, data='/kaggle/temp', out='/kaggle/working/runs'):
    """Everything after cloning: install packages, prepare the data, train all runs, report (and
    score JHMDB). max_hours is the session limit (Kaggle: 12 h); t0 = time the session started."""
    t0 = t0 or time.time()
    d = DATASETS[dataset]
    os.makedirs(data, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])
    sh(['git', '-C', REPO, 'log', '-1', '--format=code version: %h %s'])
    # Kaggle already has torch, torchvision, scipy, sklearn, opencv, h5py, pyyaml, pandas, seaborn.
    sh([sys.executable, '-m', 'pip', 'install', '-q', 'piq', 'easydict'])

    if dataset == 'calms21':
        split = prepare_calms21(data)
    else:
        split, jhmdb = prepare_jhmdb(data)

    train_hours = max_hours - (time.time() - t0) / 3600 - d['reserve']
    make_configs(runs, d['config'], split, out, train_hours)
    train(runs, d['poll'], t0)
    report(runs)
    if dataset == 'jhmdb':
        score_jhmdb(runs, split, jhmdb)
    return runs
