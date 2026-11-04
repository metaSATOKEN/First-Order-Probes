# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Bounded, streamed ZIP64 transport for PR-0004. Restored code is never run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from probe_check_io import atomic, write, read, stamp, sha, digest, safe_name, lock, evidence

EXPERIMENT='linear_fusion_pr0004_v1'
GIB=1024**3
ZIP_LIMIT=6*GIB
EXPANDED_LIMIT=16*GIB
ARCHIVE_LIMIT=40*GIB
MEMBERS_LIMIT=50000


def stream_hash(stream):
    return hashlib.file_digest(stream,'sha256').hexdigest()


def snapshot(root,destination,status):
    root,destination=Path(root),Path(destination)
    m=read(root/'manifest.json'); paths=evidence(root)
    sizes=[p.stat().st_size for p in paths]
    if len(paths)+1>MEMBERS_LIMIT or sum(sizes)>EXPANDED_LIMIT:
        raise RuntimeError('Evidence archive budget exceeded; no files dropped')
    destination.mkdir(parents=True,exist_ok=True)
    existing=sum(p.stat().st_size for p in destination.glob(EXPERIMENT+'_*.zip'))
    if existing+min(sum(sizes),ZIP_LIMIT)>ARCHIVE_LIMIT:
        raise RuntimeError('Archive cache would exceed 40 GiB; existing ZIPs are retained')
    entries={str(p.relative_to(root)):dict(size=s,sha256=digest(p)) for p,s in zip(paths,sizes)}
    index=dict(schema=1,experiment=EXPERIMENT,status=status,cell_id=m['cell_id'],run_id=m['run_id'],
               created=stamp(),entries=entries)
    target=destination/(EXPERIMENT+'_'+m['run_id']+'_'+status+'_'+uuid.uuid4().hex[:8]+'.zip')
    tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in paths: z.write(p,str(p.relative_to(root)))
        z.writestr('ARCHIVE_INDEX.json',json.dumps(index,sort_keys=True))
    if tmp.stat().st_size>ZIP_LIMIT:
        raise RuntimeError('ZIP exceeds 6 GiB; temporary archive and evidence retained')
    os.replace(tmp,target)
    return target


def inspect_zip(path,cell_id):
    if Path(path).stat().st_size>ZIP_LIMIT: raise ValueError('ZIP too large')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist(); names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>MEMBERS_LIMIT:
            raise ValueError('Duplicate/excess members')
        if sum(i.file_size for i in infos)>EXPANDED_LIMIT:
            raise ValueError('Expanded archive exceeds limit')
        for i in infos:
            safe_name(i.filename)
            if i.is_dir() or stat.S_ISLNK(i.external_attr>>16): raise ValueError('Unsafe member type')
        if z.getinfo('ARCHIVE_INDEX.json').file_size>16*1024**2:
            raise ValueError('Excess index size')
        index=json.loads(z.read('ARCHIVE_INDEX.json'))
        if (index.get('schema')!=1 or index.get('experiment')!=EXPERIMENT or index.get('cell_id')!=cell_id):
            raise ValueError('Wrong experiment / source capsule')
        if set(names)!=set(index['entries'])|{'ARCHIVE_INDEX.json'}:
            raise ValueError('Archive inventory mismatch')
        for name,item in index['entries'].items():
            if z.getinfo(name).file_size!=item['size']: raise ValueError('Member size mismatch')
            with z.open(name) as f:
                if stream_hash(f)!=item['sha256']: raise ValueError('Checksum mismatch: '+name)
        if z.getinfo('manifest.json').file_size>1024**2: raise ValueError('Manifest too large')
        m=json.loads(z.read('manifest.json'))
        if m['cell_id']!=cell_id or m['run_id']!=index['run_id']: raise ValueError('Run identity mismatch')
    return index


def choose_archive(paths,cell_id):
    if not paths: return None
    # Every candidate is checked; do not skip a damaged or mismatched candidate silently.
    pairs=[(p,inspect_zip(p,cell_id)) for p in paths]
    if len({i['run_id'] for _,i in pairs})!=1: raise ValueError('Multiple lineages: specify RESTORE_ZIP')
    return max(pairs,key=lambda x:x[1]['created'])[0]


def restore(path,root,cell_id):
    root=Path(root)
    if root.exists(): raise FileExistsError('Refuse to overwrite current run')
    index=inspect_zip(path,cell_id)
    root.parent.mkdir(parents=True,exist_ok=True)
    required=sum(v['size'] for v in index['entries'].values())+2*GIB
    if shutil.disk_usage(root.parent).free<required: raise RuntimeError('Not enough restore space')
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='restore-') as td:
        stage=Path(td)/'run'; stage.mkdir()
        with zipfile.ZipFile(path) as z:
            for name in index['entries']:
                p=stage/safe_name(name); p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,p.open('wb') as dst:
                    shutil.copyfileobj(src,dst,length=1024**2)
        os.replace(stage,root)


def new_run(root,cell_id,sources):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    write(root/'manifest.json',dict(schema=1,experiment=EXPERIMENT,cell_id=cell_id,
          run_id=uuid.uuid4().hex[:12],created=stamp(),sources={k:sha(v.encode()) for k,v in sources.items()}))
    for name,body in sources.items(): atomic(root/'sources'/safe_name(name),body.encode())


def check_run(root,cell_id,sources):
    m=read(Path(root)/'manifest.json'); expected={k:sha(v.encode()) for k,v in sources.items()}
    if m['experiment']!=EXPERIMENT or m['cell_id']!=cell_id or m['sources']!=expected:
        raise ValueError('Existing state is not this frozen capsule')
    for name,h in expected.items():
        if digest(Path(root)/'sources'/name)!=h: raise ValueError('Saved source differs')
    return m


class ActiveClock:
    """Persist elapsed active seconds at work boundaries, including failed attempts.

    Abrupt VM loss can lose the latest uncommitted interval; explicitly recorded.
    """
    def __init__(self,path,limit):
        self.path=Path(path); self.limit=limit
        self.previous=read(path)['seconds'] if self.path.exists() else 0.
        self.started=time.monotonic()
    def seconds(self): return self.previous+time.monotonic()-self.started
    def commit(self):
        write(self.path,dict(seconds=self.seconds(),at=stamp(),
              limitation='Abrupt VM loss may lose elapsed time since the last boundary.'))
    def check(self):
        self.commit()
        if self.seconds()>=self.limit: raise TimeoutError('Cumulative active budget reached')
