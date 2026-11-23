"""Train BKinD runs on a RunPod pod (any number of GPUs; one run per GPU).

    python scripts/runpod_bkind.py RUNS_FILE [--dataset calms21] [--max-hours 100] [--stop-pod]

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


def download(dataset, workspace):
    """Download (or reuse the cached copy of) the dataset's frames from Kaggle. Returns its folder."""
    os.environ.setdefault('KAGGLEHUB_CACHE', f'{workspace}/kagglehub')
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
    ap.add_argument('--no-install', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    runs = load_runs(a.runs_file)
    print('runs:', ', '.join(r['name'] for r in runs), flush=True)
    if not a.no_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + PACKAGES, check=True)
    inp = download(a.dataset, a.workspace)

    sys.path.insert(0, HERE)
    import kaggle_bkind
    try:
        kaggle_bkind.run(a.dataset, runs, max_hours=a.max_hours, t0=t0, data=a.data,
                         out=f'{a.workspace}/runs', inp=inp, runs_per_gpu=a.runs_per_gpu)
    finally:
        print(f'finished after {(time.time() - t0) / 3600:.2f} h', flush=True)
        if a.stop_pod and os.environ.get('RUNPOD_POD_ID'):
            print('stopping the pod', flush=True)
            subprocess.run(['runpodctl', 'stop', 'pod', os.environ['RUNPOD_POD_ID']])


if __name__ == '__main__':
    main()
