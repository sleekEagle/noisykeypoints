"""Train BKinD runs on a RunPod pod (any number of GPUs; one run per GPU).

    python scripts/runpod_bkind.py RUNS_FILE [--dataset calms21] [--max-hours 100] [--stop-pod] [--wait]

RUNS may list more runs than the pod has GPUs: they then train in rounds, one run per GPU at a time
(a queue). Only one launcher per pod trains at a time (a lock in /tmp); a second one exits, unless
started with --wait, in which case it waits for the first (and for any hand-started training) to end.
So several launchers started with --wait form a queue in the order they get the lock.

Usually started in the background with scripts/runpod_start.sh, so it keeps running after you close
the browser or SSH session. RUNS_FILE is a small Python file that defines RUNS, e.g.
    RUNS = [pg_ssim(98, 5000.0, 2), blur_ssim(90, 3.18, 1)]
with the helpers from scripts/run_specs.py already imported (see scripts/runpod_runs_example.py).

Folders (a RunPod network volume is mounted at /workspace and survives stopping the pod):
  /workspace/runs/<name>/   checkpoints, logs and samples (kept; a run found here resumes)
  /workspace/kagglehub/     cached download of the CalMS21 frames from the private Kaggle dataset
  /root/data/               working copy of the frames on the pod's local disk (fast; lost when the
                            pod stops, re-copied in a few minutes on the next start)
The Kaggle download needs KAGGLE_USERNAME and KAGGLE_KEY (a Kaggle API token) in the environment;
set them once as RunPod secrets / pod environment variables. Not needed once the cache exists.

--stop-pod stops the pod when all runs have ended, so it stops costing GPU money (the network
volume and its files stay). Needs RunPod's `runpodctl` and RUNPOD_POD_ID, both present on pods.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGES = ['piq', 'easydict', 'av', 'kagglehub', 'opencv-python-headless', 'h5py', 'scipy',
            'scikit-learn', 'matplotlib', 'seaborn', 'pyyaml', 'pandas']
KAGGLE_DATASETS = {'calms21': 'sleekeagle/calms21-train'}


def load_runs(path):
    sys.path.insert(0, HERE)
    import run_specs
    ns = {k: v for k, v in vars(run_specs).items() if not k.startswith('_')}
    exec(compile(open(path).read(), path, 'exec'), ns)
    assert ns.get('RUNS'), f'{path} must define a non-empty RUNS list'
    names = [r['name'] for r in ns['RUNS']]
    assert len(set(names)) == len(names), f'run names must be unique: {names}'
    return ns['RUNS']


def acquire_pod_lock(wait, path='/tmp/runpod_bkind.lock'):
    """Exclusive lock for this pod. Without --wait, exit if another launcher holds it; with --wait,
    block until it is free. The lock is released when this process exits (even if it crashes)."""
    import fcntl
    f = open(path, 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not wait:
            raise SystemExit('another launcher is training on this pod; start with --wait to queue behind it')
        print(time.strftime('%H:%M'), 'waiting for the launcher that is running now to finish', flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
        print(time.strftime('%H:%M'), 'previous launcher finished; starting', flush=True)
    return f


def download(dataset, workspace):
    """Download (or reuse the cached copy of) the dataset's frames from Kaggle. Returns its folder.
    A finished earlier download (one with videos.json) is used directly, without asking Kaggle, so
    the Kaggle token is only needed the first time."""
    os.environ.setdefault('KAGGLEHUB_CACHE', f'{workspace}/kagglehub')
    cached = os.path.join(os.environ['KAGGLEHUB_CACHE'], 'datasets', *KAGGLE_DATASETS[dataset].split('/'))
    versions = [v for v in glob.glob(f'{cached}/versions/*')
                if os.path.basename(v).isdigit() and glob.glob(f'{v}/**/videos.json', recursive=True)]
    if versions:
        path = max(versions, key=lambda v: int(os.path.basename(v)))  # newest finished version
        print(f'{dataset} frames: using the cached download {path}')
        return path
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASETS[dataset])
    print(f'{dataset} frames: {path}')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs_file')
    ap.add_argument('--dataset', default='calms21', choices=['calms21'])
    ap.add_argument('--max-hours', type=float, default=100.0,
                    help='stop cleanly between epochs after this long (a cost cap; RunPod has no session limit)')
    ap.add_argument('--runs-per-gpu', type=int, default=1)
    ap.add_argument('--workspace', default='/workspace')
    ap.add_argument('--data', default='/root/data', help='local working copy of the frames')
    ap.add_argument('--stop-pod', action='store_true', help='stop this pod when training has ended')
    ap.add_argument('--wait', action='store_true',
                    help='first wait until no other training (train_video.py) is running on this pod')
    ap.add_argument('--no-install', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    # RunPod's newer images mark the system Python as "externally managed" (PEP 668), so pip refuses
    # to install into it. The pod's container is disposable, so allow it (also for kaggle_bkind's pip).
    os.environ['PIP_BREAK_SYSTEM_PACKAGES'] = '1'
    runs = load_runs(a.runs_file)
    print('runs:', ', '.join(r['name'] for r in runs), flush=True)
    if not a.no_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + PACKAGES, check=True)
    inp = download(a.dataset, a.workspace)

    # Only one launcher per pod uses the GPU at a time: a lock on the pod's local disk (not on the
    # network volume, which other pods share). --wait queues behind the launcher holding it.
    lock = acquire_pod_lock(a.wait)  # noqa: F841 (held until this process exits)

    if a.wait:  # also wait for training started outside a launcher (e.g. a hand-started test)
        while subprocess.run(['pgrep', '-f', 'python.*train_video.py'], capture_output=True).returncode == 0:
            print(time.strftime('%H:%M'), 'waiting: another training is still running', flush=True)
            time.sleep(60)

    sys.path.insert(0, HERE)
    import kaggle_bkind
    import torch
    slots = max(1, torch.cuda.device_count()) * a.runs_per_gpu
    batches = [runs[i:i + slots] for i in range(0, len(runs), slots)]
    if len(batches) > 1:
        print(f'{len(runs)} runs, {slots} at a time: {len(batches)} rounds', flush=True)
    try:
        for i, batch in enumerate(batches):  # a queue: more runs than GPUs train in turns
            if len(batches) > 1:
                print(f'===== round {i + 1}/{len(batches)}:', ', '.join(r['name'] for r in batch), flush=True)
            kaggle_bkind.run(a.dataset, batch, max_hours=a.max_hours, t0=t0, data=a.data,
                             out=f'{a.workspace}/runs', inp=inp, runs_per_gpu=a.runs_per_gpu)
    finally:
        print(f'finished after {(time.time() - t0) / 3600:.2f} h', flush=True)
        if a.stop_pod and os.environ.get('RUNPOD_POD_ID'):
            print('stopping the pod', flush=True)
            subprocess.run(['runpodctl', 'stop', 'pod', os.environ['RUNPOD_POD_ID']])


if __name__ == '__main__':
    main()
