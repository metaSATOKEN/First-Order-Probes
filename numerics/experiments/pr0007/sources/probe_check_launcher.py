# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Colab-only launcher. Restored evidence is data, never executable code."""
import importlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

import probe_check_io as disk

PINS = {'numpy': '2.1.3', 'scipy': '1.16.3', 'cupy-cuda12x': '14.2.0'}


def child_environment():
    env = os.environ.copy()
    # Keep CUDA loader paths. Do not inherit notebook Python injection paths.
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    return env


def stop_owned(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def stream_process(command, timeout, log, on_checkpoint=None):
    """Drain stdout in a thread, with deadlines and bounded retained output."""
    start = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=child_environment(), start_new_session=True)
    messages = queue.Queue()
    def reader():
        for line in proc.stdout:
            messages.put(line)
        messages.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    retained = ''
    try:
        eof = False
        while not eof or proc.poll() is None:
            if time.monotonic()-start > timeout:
                raise TimeoutError('Subprocess budget exceeded')
            try:
                line = messages.get(timeout=.5)
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            print(line, end='', flush=True)
            retained = (retained + line)[-1024*1024:]
            # Log lives outside the run while the worker makes snapshots.
            disk.atomic(log, retained.encode())
            if line.startswith('CHECKPOINT_ZIP=') and on_checkpoint:
                on_checkpoint(Path(line.strip().split('=',1)[1]))
        return proc.wait()
    finally:
        stop_owned(proc)
        thread.join(timeout=2)
        proc.stdout.close()
        disk.atomic(log, retained.encode())


def resolve_run(base, cell_id, sources, restore_zip=''):
    root = base/'runs'/('current_'+cell_id)
    if root.exists():
        if restore_zip:
            raise ValueError('Current runtime already has this run. Clear RESTORE_ZIP and rerun; no overwrite.')
        disk.check_run(root,cell_id,sources)
        print('Existing state found: verified, automatic resume.',flush=True)
        return root, 'existing'
    if restore_zip=='UPLOAD':
        from google.colab import files
        uploaded = files.upload()
        paths = [Path(name).resolve() for name in uploaded]
        if len(paths)!=1:
            raise ValueError('Upload exactly one pilot result ZIP')
        selected = paths[0]
    elif restore_zip:
        selected = Path(restore_zip).expanduser().resolve()
    else:
        paths = list(Path('/content').glob('probe_cpu_gpu_check_v1_*.zip')) + list(base.glob('probe_cpu_gpu_check_v1_*.zip'))
        selected = disk.choose_archive(paths,cell_id)
    if selected:
        disk.restore(selected,root,cell_id)
        disk.check_run(root,cell_id,sources)
        print('Restored and verified: '+str(selected),flush=True)
        return root, 'restored'
    disk.new_run(root,cell_id,sources)
    print('Fresh GPU pilot: no upload or previous receipt required.',flush=True)
    return root, 'fresh'


def ensure_environment(base, cell_id, logs):
    envdir = base/('env_'+cell_id)
    python = envdir/'bin/python'
    started = time.monotonic()
    if not python.exists():
        subprocess.run([sys.executable,'-m','venv','--without-pip',str(envdir)],
                       env=child_environment(), check=True,timeout=120)
    code = ('import importlib.metadata as m,json; p='+repr(PINS)+'; '
            'v={k:m.version(k) for k in p}; print(json.dumps(v)); assert v==p')
    probe = subprocess.run([str(python),'-c',code], capture_output=True,text=True,
                           env=child_environment(),timeout=60)
    if probe.returncode:
        cmd = [sys.executable,'-m','pip','--python',str(python),'install',
               '--disable-pip-version-check','--only-binary=:all:',
               'numpy==2.1.3','scipy==1.16.3','cupy-cuda12x[ctk]==14.2.0']
        budget = max(1,1200-(time.monotonic()-started))
        rc = stream_process(cmd,budget,logs/'install.log')
        if rc:
            raise RuntimeError(f'Isolated dependency installation failed (exit {rc})')
    probe = subprocess.run([str(python),'-c',code],capture_output=True,text=True,
                           env=child_environment(),timeout=60,check=True)
    return python, dict(seconds=time.monotonic()-started, pins=json.loads(probe.stdout),
                        isolated_prefix=str(envdir), python_injection_removed=True,
                        cuda_loader_environment_preserved=True)


def run(sources, cell_id, package, restore_zip='', auto_download=True):
    if sys.platform!='linux' or not Path('/content').is_dir():
        raise RuntimeError('Paste this cell into a new Colab GPU runtime. Local benchmark is disabled.')
    importlib.import_module('google.colab')
    base = Path('/content/probe_cpu_gpu_check_v1')
    base.mkdir(exist_ok=True)
    with disk.lock(base/'run.lock'):
        root, mode = resolve_run(base,cell_id,sources,restore_zip)
        launch_id = uuid.uuid4().hex[:12]
        logs = base/'launcher_logs'/launch_id
        logs.mkdir(parents=True)
        started = time.monotonic()
        status, rc = 'TECHNICAL_FAILURE', None
        downloads = []
        def download(path):
            # A browser can reject automatic downloads; the file remains in Files.
            entry = dict(path=str(path),time=disk.stamp(),requested=bool(auto_download))
            if auto_download:
                try:
                    from google.colab import files
                    files.download(str(path))
                    entry['request_sent'] = True
                except Exception as exc:
                    entry.update(request_sent=False,error=str(exc))
                    print('Automatic download could not start. Use Colab Files: '+str(path),flush=True)
            downloads.append(entry)
        try:
            free = shutil.disk_usage(base).free
            if mode=='fresh' and free<8*1024**3:
                raise RuntimeError('Fresh pilot requires at least 8 GiB free disk')
            try:
                smi = subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
                disk.atomic(root/'launches'/launch_id/'nvidia-smi.txt',(smi.stdout+smi.stderr).encode())
                if smi.returncode:
                    raise RuntimeError('nvidia-smi failed: choose a GPU runtime')
            except FileNotFoundError as exc:
                raise RuntimeError('No NVIDIA GPU found: choose a Colab GPU runtime') from exc
            python, info = ensure_environment(base,cell_id,logs)
            disk.write(root/'launches'/launch_id/'bootstrap.json',dict(mode=mode,free_bytes=free,**info))
            rc = stream_process([str(python),'-u',str(Path(package)/'probe_backend_check_v1.py'),str(root)],
                                3720,logs/'worker.log',on_checkpoint=download)
            summary = disk.read(root/'summary.json') if (root/'summary.json').exists() else {}
            status = summary.get('status','TECHNICAL_FAILURE') if rc in (0,2) else 'TECHNICAL_FAILURE'
            if rc==0 and status!='PILOT_COMPLETE':
                raise RuntimeError('Worker exit and summary status disagree')
        except BaseException as exc:
            disk.write(root/'launches'/launch_id/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            print(traceback.format_exc(),flush=True)
        finally:
            # Child has exited before any launcher writes enter the snapshot.
            for path in logs.glob('*.log'):
                disk.atomic(root/'launches'/launch_id/path.name,path.read_bytes())
            disk.write(root/'launches'/launch_id/'launch.json',dict(status=status,returncode=rc,mode=mode,
                elapsed_seconds=time.monotonic()-started,checkpoint_downloads=downloads,
                final_download='requested after this manifest; browser completion is not observable'))
            try:
                final = disk.snapshot(root,base,status)
                disk.inspect_zip(final,cell_id)
                print('\nResult ZIP: '+str(final)+'\nStatus: '+status,flush=True)
                download(final)
            except Exception as exc:
                print('ZIP recovery failed; original files remain at '+str(root)+': '+str(exc),flush=True)
                raise
        if status!='PILOT_COMPLETE':
            print('Not a GPU acceptance pass. Keep the ZIP and share it for diagnosis; no automatic precision/budget changes.',flush=True)
        return dict(status=status,root=str(root),zip=str(final))
