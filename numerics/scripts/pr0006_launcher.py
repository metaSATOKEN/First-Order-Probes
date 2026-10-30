# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Drive-free Colab launcher. Only embedded trusted sources are executable."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import pr0006_io as disk
from probe_check_launcher import ensure_environment,stream_process,child_environment


def resolve_run(base,cell_id,sources,restore_zip='',content=Path('/content'),upload=None):
    base=Path(base);root=base/'runs'/('current_'+cell_id)
    if root.exists():
        disk.check_run(root,cell_id,sources);disk.verify_carry(root)
        print('Existing runtime: verified automatic resume; no ZIP selection needed.',flush=True)
        return root,'existing'
    if restore_zip=='UPLOAD':
        if upload is None:
            from google.colab import files
            upload=files.upload
        paths=[Path(n).resolve() for n in upload()]
        if len(paths)!=1:raise ValueError('Choose ONE latest checkpoint or final index; retain other data ZIPs locally')
        selected=paths[0]
    elif restore_zip:selected=Path(restore_zip).expanduser().resolve()
    else:
        selected=disk.choose_archive(list(Path(content).glob(disk.EXPERIMENT+'_*.zip'))+
                                     list((base/'archives').glob(disk.EXPERIMENT+'_*.zip')),cell_id)
    if selected:
        remote=disk.restore(selected,root,cell_id,sources)
        print(f'Restored latest checkpoint; {len(remote)} prior replicates are archive-certified (raw not re-read here).',flush=True)
        return root,'restored'
    if shutil.disk_usage(base).free<24*1024**3:raise RuntimeError('Fresh run needs 24 GiB free disk')
    disk.new_run(root,cell_id,sources)
    print('Fresh B25 run: no Drive, no input ZIP, no restart selector.',flush=True)
    return root,'fresh'


def recovery_reps(root):
    reps=[]
    for shard in (Path(root)/'shards').glob('[0-9][0-9]'):
        for folder in shard.glob('rep_[0-9][0-9]'):
            r=int(folder.name[4:])
            if folder==disk.rep_dir(root,r) and 0<=r<25:reps.append(r)
    if not reps:return []
    latest=max(reps)//5
    return sorted(r for r in reps if r//5==latest)


def archive_job(root,archives,status,receipt):
    reps=[] if status=='COMPLETE_INDEX' else recovery_reps(root)
    paths=disk.snapshot(root,archives,status,reps)
    disk.write(receipt,dict(paths=[str(p) for p in paths]))


def run(sources,cell_id,package,restore_zip='',auto_download=True):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Use a Colab GPU runtime; local main disabled')
    importlib.import_module('google.colab')
    base=Path('/content')/disk.EXPERIMENT;base.mkdir(exist_ok=True);package=Path(package)
    with disk.lock(base/'run.lock'):
        root,mode=resolve_run(base,cell_id,sources,restore_zip)
        session=uuid.uuid4().hex[:12];logs=base/'launcher_logs'/session;logs.mkdir(parents=True)
        status='TECHNICAL_FAILURE';rc=None;downloads=[];archive_created=False;started=time.monotonic()
        def download(path):
            if path.stat().st_size>=disk.ZIP_LIMIT:raise ValueError('Refuse download >= decimal 1 GB')
            item=dict(path=str(path),bytes=path.stat().st_size,at=disk.stamp(),requested=bool(auto_download))
            if auto_download:
                try:
                    from google.colab import files
                    files.download(str(path));item['request_sent']=True
                except Exception as exc:
                    item.update(request_sent=False,error=str(exc));print('Download could not start; file retained: '+str(path),flush=True)
            downloads.append(item)
        try:
            if len(disk.read(root/'state.json')['completed'])<25:
                smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30,check=True)
                disk.atomic(root/'sessions'/session/'nvidia-smi.txt',smi.stdout.encode())
            python,info=ensure_environment(base,cell_id,logs)
            disk.write(root/'sessions'/session/'bootstrap.json',dict(mode=mode,**info))
            elapsed=disk.read(root/'active_time.json')['seconds'] if (root/'active_time.json').exists() else 0.
            rc=stream_process([str(python),'-u',str(package/'pr0006_runner.py'),str(root),str(base/'archives'),session],
                              max(120,14400-elapsed+120),logs/'worker.log',on_checkpoint=download)
            if rc==0 and disk.read(root/'state.json')['status']=='COMPLETE':status='COMPLETE_INDEX'
            elif rc==3:status='PARTIAL'
        except BaseException as exc:
            status='PARTIAL' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'TECHNICAL_FAILURE'
            disk.write(root/'sessions'/session/'launcher_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            print(traceback.format_exc(),flush=True)
        finally:
            # Owned worker has stopped before snapshot: no mixed-time evidence tree.
            for p in logs.glob('*.log'):disk.atomic(root/'sessions'/session/p.name,p.read_bytes())
            disk.write(root/'sessions'/session/'launch.json',dict(mode=mode,status=status,returncode=rc,
                       wall_seconds=time.monotonic()-started,downloads=downloads,
                       note='Download request is not proof of browser completion; all server files are retained.'))
            receipt=logs/'archive_receipt.json'
            try:
                cmd=[sys.executable,str(package/'pr0006_launcher.py'),'archive',str(root),str(base/'archives'),status,str(receipt)]
                subprocess.run(cmd,env=child_environment(),check=True,timeout=1800)
                for p in disk.read(receipt)['paths']:download(Path(p))
                archive_created=True
            except BaseException as exc:
                disk.write(root/'sessions'/session/'archive_error.json',dict(type=type(exc).__name__,message=str(exc)))
                print('Archive delivery failed; saved runtime data remain at '+str(root),flush=True)
                print(str(exc),flush=True)
        return dict(status='COMPLETE' if status=='COMPLETE_INDEX' else status,run=str(root),
                    archives=str(base/'archives'),downloads=downloads,final_archive_created=archive_created,
                    reminder='Keep all five data shards (or fallback parts) plus the small final index. No science was rerun for archiving.')


if __name__=='__main__':
    if len(sys.argv)!=6 or sys.argv[1]!='archive':raise SystemExit('Archive helper only')
    archive_job(*sys.argv[2:])
