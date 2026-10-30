# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Drive-free launcher; trusted embedded code only, never restored executable code."""
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import pr0012_io as disk
from probe_check_launcher import ensure_environment,stream_process,child_environment
from pr0012_download import Delivery,upload_streaming


def resolve_run(base,cell_id,sources,restore_zips='',content=Path('/content'),upload=None):
    base=Path(base);root=base/'runs'/('current_'+cell_id)
    if root.exists():
        disk.check_run(root,cell_id,sources);disk.verify_carry(root)
        print('Existing runtime: verified auto-resume, no ZIP picker.',flush=True);return root,'existing'
    if restore_zips=='UPLOAD':
        paths=(upload or upload_streaming)(base/'uploads')
    elif restore_zips:
        paths=[Path(p).expanduser().resolve() for p in (restore_zips if isinstance(restore_zips,list) else [restore_zips])]
    else:
        paths=list(Path(content).glob(disk.EXPERIMENT+'_*.zip'))+list((base/'archives').glob('*.zip'))+list((base/'uploads').glob('*.zip'))
    if paths:
        began=time.monotonic();remote=disk.restore_many(paths,root,cell_id,sources)
        if time.monotonic()-began>1800:raise TimeoutError('Restore exceeded 30 minutes; restored files retained')
        print(f'Restored: {len(remote)} completed replicates receipt-certified, raw not present/re-read here.',flush=True)
        return root,'restored'
    if restore_zips:raise ValueError('No restore ZIP selected; will not silently start fresh')
    if shutil.disk_usage(base).free<24*1024**3:raise RuntimeError('Fresh run needs 24GiB free disk')
    disk.new_run(root,cell_id,sources);print('Fresh B50: no Drive, no input ZIP, no 45-minute stop.',flush=True)
    return root,'fresh'


def recovery_reps(root):
    reps=[r for r in range(50) if disk.rep_dir(root,r).exists()]
    return [r for r in reps if r//5==max(reps)//5] if reps else []


def archive_job(root,archives,status,receipt):
    paths=disk.snapshot(root,archives,status,[] if status=='COMPLETE_INDEX' else recovery_reps(root))
    disk.write(receipt,dict(paths=[str(p) for p in paths]))


def run(sources,cell_id,package,restore_zips='',auto_download=True):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Use Colab GPU; local scientific main disabled')
    importlib.import_module('google.colab');base=Path('/content')/disk.EXPERIMENT;base.mkdir(exist_ok=True)
    package=Path(package)
    with disk.lock(base/'run.lock'):
        root,mode=resolve_run(base,cell_id,sources,restore_zips)
        session=uuid.uuid4().hex[:12];logs=base/'launcher_logs'/session;logs.mkdir(parents=True)
        delivery_folder=base/'delivery'/disk.read(root/'manifest.json')['run_id']
        try:delivery=Delivery(delivery_folder,auto_download)
        except Exception as exc:
            print('Browser queue setup unavailable; computation and disk checkpoints continue: '+str(exc),flush=True)
            delivery=Delivery(delivery_folder,False)
        status='TECHNICAL_FAILURE';rc=None;archive_created=False;started=time.monotonic()
        def download(path):
            try:delivery.enqueue(path)
            except Exception as exc:print('Transfer queue unavailable; science continues; ZIP retained: '+str(path)+' | '+str(exc),flush=True)
        # Retry only archives not acknowledged as received/save-requested.
        for a in disk.catalog(root)['archives']:
            p=base/'archives'/a['name']
            if p.exists():download(p)
        if disk.read(root/'state.json')['status']=='COMPLETE' and any(
                a['status']=='COMPLETE_INDEX' for a in disk.catalog(root)['archives']):
            print('COMPLETE verified: no new scientific work or re-packing. Pending downloads requeued.',flush=True)
            return dict(status='COMPLETE_NOOP',run=str(root),archives=str(base/'archives'),
                        reminder='Browser save requests are not disk confirmation.')
        try:
            if len(disk.read(root/'state.json')['completed'])<50:
                smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30,check=True)
                disk.atomic(root/'sessions'/session/'nvidia-smi.txt',smi.stdout.encode())
            python,info=ensure_environment(base,cell_id,logs)
            disk.write(root/'sessions'/session/'bootstrap.json',dict(mode=mode,**info))
            elapsed=disk.read(root/'active_time.json')['seconds'] if (root/'active_time.json').exists() else 0.
            rc=stream_process([str(python),'-u',str(package/'pr0012_runner.py'),str(root),str(base/'archives'),session],
                max(120,28800-elapsed+120),logs/'worker.log',on_checkpoint=download)
            if rc==0 and disk.read(root/'state.json')['status']=='COMPLETE':status='COMPLETE_INDEX'
            elif rc==3:status='PARTIAL'
        except BaseException as exc:
            status='PARTIAL' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'TECHNICAL_FAILURE'
            disk.write(root/'sessions'/session/'launcher_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            print(traceback.format_exc(),flush=True)
        finally:
            for p in logs.glob('*.log'):disk.atomic(root/'sessions'/session/p.name,p.read_bytes())
            disk.write(root/'sessions'/session/'launch.json',dict(mode=mode,status=status,returncode=rc,wall_seconds=time.monotonic()-started))
            disk.write(root/'delivery_status.json',delivery.status())
            receipt=logs/'archive_receipt.json'
            try:
                subprocess.run([sys.executable,str(package/'pr0012_launcher.py'),'archive',str(root),str(base/'archives'),status,str(receipt)],
                    env=child_environment(),check=True,timeout=1800)
                for p in disk.read(receipt)['paths']:download(Path(p))
                archive_created=True
            except BaseException as exc:
                disk.write(root/'sessions'/session/'archive_error.json',dict(type=type(exc).__name__,message=str(exc)))
                print('Archive failed; original saved data retained: '+str(root)+' | '+str(exc),flush=True)
        return dict(status='COMPLETE' if status=='COMPLETE_INDEX' else status,run=str(root),archives=str(base/'archives'),
            final_archive_created=archive_created,delivery_status=str(delivery.folder),
            reminder='Keep ten data shards (plus any partial/fallback parts) and final index. Browser queue may still be running; disk save is not asserted.')


if __name__=='__main__':
    if len(sys.argv)!=6 or sys.argv[1]!='archive':raise SystemExit('Archive helper only')
    archive_job(*sys.argv[2:])
