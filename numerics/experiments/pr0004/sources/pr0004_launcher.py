# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Drive-free Colab launcher, sharing tested process/venv helpers with PR-0003."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
import pr0004_io as disk
from probe_check_launcher import ensure_environment,stream_process


def resolve_run(base,cell_id,sources,restore_zip=''):
    root=base/'runs'/('current_'+cell_id)
    if root.exists():
        if restore_zip: raise ValueError('Run already exists: leave RESTORE_ZIP blank; no overwrite')
        disk.check_run(root,cell_id,sources)
        print('Verified current runtime: automatic resume.',flush=True)
        return root,'existing'
    if restore_zip=='UPLOAD':
        from google.colab import files
        uploaded=files.upload(); names=list(uploaded); del uploaded
        if len(names)!=1: raise ValueError('Select exactly one PR-0004 ZIP')
        selected=Path(names[0]).resolve()
    elif restore_zip:
        selected=Path(restore_zip).expanduser().resolve()
    else:
        paths=list(Path('/content').glob(disk.EXPERIMENT+'_*.zip'))+list(base.glob(disk.EXPERIMENT+'_*.zip'))
        selected=disk.choose_archive(paths,cell_id)
    if selected:
        disk.restore(selected,root,cell_id); disk.check_run(root,cell_id,sources)
        print('Restored verified ZIP: '+str(selected),flush=True)
        return root,'restored'
    if shutil.disk_usage(base).free<64*disk.GIB:
        raise RuntimeError('A fresh full run needs at least 64 GiB free disk; no prior ZIPs are deleted')
    disk.new_run(root,cell_id,sources)
    print('Fresh PR-0004 main run: no upload required.',flush=True)
    return root,'fresh'


def run(sources,cell_id,package,restore_zip='',auto_download=True):
    if sys.platform!='linux' or not Path('/content').is_dir():
        raise RuntimeError('Run this cell in Colab GPU; local main execution is disabled')
    importlib.import_module('google.colab')
    base=Path('/content')/disk.EXPERIMENT; base.mkdir(exist_ok=True)
    with disk.lock(base/'run.lock'):
        root,mode=resolve_run(base,cell_id,sources,restore_zip)
        launch=uuid.uuid4().hex[:12]; logs=base/'launcher_logs'/launch; logs.mkdir(parents=True)
        began=time.monotonic(); status='TECHNICAL_FAILURE'; rc=None; downloads=[]
        def download(path):
            row=dict(path=str(path),at=disk.stamp(),requested=bool(auto_download))
            if auto_download:
                try:
                    from google.colab import files
                    files.download(str(path)); row['request_sent']=True
                except Exception as exc:
                    row.update(request_sent=False,error=str(exc))
                    print('Download did not start. Use Colab Files: '+str(path),flush=True)
            downloads.append(row)
        try:
            smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            disk.atomic(root/'launches'/launch/'nvidia-smi.txt',(smi.stdout+smi.stderr).encode())
            if smi.returncode: raise RuntimeError('Choose a GPU runtime; CPU fallback is not enabled')
            python,info=ensure_environment(base,cell_id,logs)
            disk.write(root/'launches'/launch/'bootstrap.json',dict(mode=mode,**info))
            spent=disk.read(root/'active_time.json')['seconds'] if (root/'active_time.json').exists() else 0.
            remaining=max(0.,21600-spent)
            if remaining<=0: raise TimeoutError('Six-hour cumulative worker budget exhausted; no automatic reset')
            rc=stream_process([str(python),'-u',str(Path(package)/'pr0004_runner.py'),str(root)],
                remaining+120,logs/'worker.log',on_checkpoint=download)
            summary=disk.read(root/'summary.json') if (root/'summary.json').exists() else {}
            status=summary.get('status','TECHNICAL_FAILURE') if rc in (0,2) else 'TECHNICAL_FAILURE'
            if rc==0 and status!='COMPLETE': raise RuntimeError('Exit/summary mismatch')
        except BaseException as exc:
            status='PARTIAL' if isinstance(exc,(KeyboardInterrupt,TimeoutError)) else 'TECHNICAL_FAILURE'
            disk.write(root/'launches'/launch/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            print(traceback.format_exc(),flush=True)
        finally:
            for path in logs.glob('*.log'): disk.atomic(root/'launches'/launch/path.name,path.read_bytes())
            disk.write(root/'launches'/launch/'launch.json',dict(status=status,returncode=rc,mode=mode,
                elapsed_seconds_before_final_zip=time.monotonic()-began,checkpoint_downloads=downloads,
                final_download='Requested after archive; browser completion is not observable'))
            # In a child process so the 30-minute archive recovery deadline is enforceable.
            receipt=logs/'archive_receipt.json'
            command=[sys.executable,str(Path(package)/'pr0004_launcher.py'),'archive',str(root),str(base),status,cell_id,str(receipt)]
            archive_rc=stream_process(command,1800,logs/'archive.log')
            if archive_rc or not receipt.exists():
                raise RuntimeError('Final ZIP recovery failed; evidence remains at '+str(root))
            record=disk.read(receipt); final=Path(record['path'])
            print('Result ZIP: '+str(final)+'\nStatus: '+status+'\nArchive seconds: '+str(record['seconds']),flush=True)
            download(final)
        return dict(status=status,root=str(root),zip=str(final))


if __name__=='__main__':
    if len(sys.argv)!=7 or sys.argv[1]!='archive': raise SystemExit('Launcher is called by the Colab capsule')
    _,_,root,base,status,cell_id,receipt=sys.argv
    start=time.monotonic(); path=disk.snapshot(root,base,status); created=time.monotonic()-start
    disk.inspect_zip(path,cell_id); verified=time.monotonic()-start-created
    # ZIP comments carry transport timings after the indexed evidence was closed.
    # No recursive inclusion of the ZIP's own hash or post-download status.
    with zipfile.ZipFile(path,'a') as z:
        z.comment=json.dumps(dict(archive_creation_seconds=created,archive_verification_seconds=verified,
                                  browser_download_completion='not observable')).encode()
    disk.write(receipt,dict(path=str(path),seconds=time.monotonic()-start,size=path.stat().st_size,sha256=disk.digest(path)))
