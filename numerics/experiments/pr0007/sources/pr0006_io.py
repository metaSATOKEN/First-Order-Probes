# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Non-cumulative bounded transport. Restored sources are never executed."""
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
from probe_check_io import atomic,write,read,stamp,sha,digest,safe_name,lock,evidence

EXPERIMENT='lfdc_budget_pr0006_v1'
ZIP_LIMIT=1_000_000_000
PREPACK_LIMIT=800_000_000
MEMBERS_LIMIT=20000
CACHE_LIMIT=16*1024**3
ROOT_FILES={'manifest.json','specification.json','state.json','active_time.json','catalog.json','summary.json'}
ROOT_DIRS={'sources','carry','sessions','shards','analysis'}


def allowed(name):
    p=safe_name(name)
    if len(p.parts)==1 and name in ROOT_FILES:return p
    if len(p.parts)>1 and p.parts[0] in ROOT_DIRS:return p
    raise ValueError('Unexpected evidence path: '+name)


def new_run(root,cell_id,sources):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write(root/'manifest.json',dict(schema=1,experiment=EXPERIMENT,cell_id=cell_id,run_id=uuid.uuid4().hex[:12],
          created=stamp(),sources={n:sha(b.encode()) for n,b in sources.items()}))
    for name,body in sources.items():
        if Path(name).name!=name:raise ValueError('Source capsule name must be flat')
        atomic(root/'sources'/name,body.encode())
    write(root/'state.json',dict(completed={},status='NEW'))
    write(root/'catalog.json',dict(archives=[]))


def check_run(root,cell_id,sources):
    root=Path(root);m=read(root/'manifest.json');expected={n:sha(b.encode()) for n,b in sources.items()}
    if m.get('experiment')!=EXPERIMENT or m.get('cell_id')!=cell_id or m['sources']!=expected:
        raise ValueError('Run/source capsule mismatch')
    for n,h in expected.items():
        if digest(root/'sources'/n)!=h:raise ValueError('Saved source mismatch: '+n)
    return m


def rep_dir(root,rep):return Path(root)/'shards'/f'{rep//5:02d}'/f'rep_{rep:02d}'


def catalog(root):
    p=Path(root)/'catalog.json'
    return read(p) if p.exists() else dict(archives=[])


def verify_carry(root):
    root=Path(root);state=read(root/'state.json');remote=[]
    cat=catalog(root)['archives']
    ids=sorted(int(k) for k in state['completed'])
    if ids!=list(range(len(ids))):raise ValueError('Completed replicates must be a contiguous prefix')
    for key,record in state['completed'].items():
        rep=int(key)
        if key!=f'{rep:02d}' or not 0<=rep<25:raise ValueError('Unexpected replicate ID')
        meta=root/'carry'/f'rep_{rep:02d}.json';data=meta.with_suffix('.npz')
        if digest(meta)!=record['metadata_sha256']:raise ValueError('Carry metadata mismatch')
        d=read(meta)
        if d['replicate']!=rep or d['records']!=549988 or digest(data)!=d['statistics_sha256']:
            raise ValueError('Carry identity/statistics mismatch')
        folder=rep_dir(root,rep)
        if folder.exists():
            found={str(p.relative_to(folder)) for p in evidence(folder)}
            if found!=set(d['files']):raise ValueError('Completed replicate inventory differs')
            for name,h in d['files'].items():
                if digest(folder/safe_name(name))!=h:raise ValueError('Completed evidence changed')
        else:
            receipts=[x for x in cat if x.get('carry_hashes',{}).get(key)==record['metadata_sha256'] and rep in x.get('covered_replicates',[])]
            if not receipts:raise ValueError('Missing prior evidence with no verified archive receipt')
            remote.append(rep)
    return remote


def select_paths(root,reps):
    root=Path(root);out=[]
    for p in evidence(root):
        rel=p.relative_to(root)
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Symlink evidence rejected')
        allowed(str(rel))
        if rel.parts[0]=='shards':
            if len(rel.parts)<4:raise ValueError('Invalid shard tree')
            if not any(p.is_relative_to(rep_dir(root,r)) for r in reps):continue
        out.append(p)
    return out


def snapshot_one(root,destination,status,reps):
    root,destination=Path(root),Path(destination);m=read(root/'manifest.json')
    requested=list(reps);paths=select_paths(root,requested)
    # After per-replicate fallback restoration, earlier raw in this same shard
    # may be remote. Never issue a new raw-coverage receipt for absent files.
    reps=[r for r in requested if any(p.is_relative_to(rep_dir(root,r)) for p in paths)]
    raw=sum(p.stat().st_size for p in paths)
    if raw>PREPACK_LIMIT or len(paths)+1>MEMBERS_LIMIT:raise OverflowError('Prepack budget exceeded')
    destination.mkdir(parents=True,exist_ok=True)
    if sum(p.stat().st_size for p in destination.glob('*.zip'))+raw+16*1024**2>CACHE_LIMIT:
        raise RuntimeError('ZIP cache budget reached; old evidence retained')
    entries={str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=digest(p)) for p in paths}
    state=read(root/'state.json')
    index=dict(schema=1,experiment=EXPERIMENT,status=status,cell_id=m['cell_id'],run_id=m['run_id'],created=stamp(),
               covered_replicates=list(reps),requested_replicates=requested,
               carry_hashes={k:v['metadata_sha256'] for k,v in state['completed'].items()},entries=entries)
    name=EXPERIMENT+'_'+m['run_id']+'_'+status+'_'+uuid.uuid4().hex[:8]+'.zip';target=destination/name;tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in paths:z.write(p,str(p.relative_to(root)))
        z.writestr('ARCHIVE_INDEX.json',json.dumps(index,sort_keys=True))
    if tmp.stat().st_size>=ZIP_LIMIT:raise OverflowError('Final ZIP must be strictly below decimal 1 GB; temporary file retained')
    os.replace(tmp,target);inspect_zip(target,m['cell_id'])
    record=dict(name=name,sha256=digest(target),bytes=target.stat().st_size,created=index['created'],
        status=status,covered_replicates=list(reps),carry_hashes={k:v for k,v in index['carry_hashes'].items() if int(k) in reps})
    c=catalog(root);c['archives'].append(record);write(root/'catalog.json',c)
    return target


def snapshot(root,destination,status,reps):
    """Only selected <=5 replicates plus small carry metadata; never all old raw."""
    reps=sorted(set(reps))
    if len(reps)>5:raise ValueError('Snapshot spans more than one small experiment')
    if len({r//5 for r in reps})>1:raise ValueError('Snapshot crosses small-experiment boundary')
    try:return [snapshot_one(root,destination,status,reps)]
    except OverflowError:
        if len(reps)<2:raise
        # Fixed, scientific-ID-preserving fallback. Latest part is the recovery entry.
        return [snapshot_one(root,destination,status+'_REP'+f'{r:02d}',[r]) for r in reps]


def inspect_zip(path,cell_id):
    if Path(path).stat().st_size>=ZIP_LIMIT:raise ValueError('ZIP >= decimal 1 GB')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[x.filename for x in infos]
        if len(names)!=len(set(names)) or len(names)>MEMBERS_LIMIT:raise ValueError('Duplicate/excess archive members')
        if sum(x.file_size for x in infos)>PREPACK_LIMIT+16*1024**2:raise ValueError('Excess expanded archive')
        for i in infos:
            safe_name(i.filename)
            if i.is_dir() or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe member type')
            if i.filename!='ARCHIVE_INDEX.json':allowed(i.filename)
        if z.getinfo('ARCHIVE_INDEX.json').file_size>16*1024**2:raise ValueError('Excess index')
        index=json.loads(z.read('ARCHIVE_INDEX.json'))
        if index.get('schema')!=1 or index.get('experiment')!=EXPERIMENT or index.get('cell_id')!=cell_id:
            raise ValueError('Wrong experiment/cell')
        if set(names)!=set(index['entries'])|{'ARCHIVE_INDEX.json'}:raise ValueError('Inventory mismatch')
        for name,e in index['entries'].items():
            if z.getinfo(name).file_size!=e['size']:raise ValueError('Member size mismatch')
            with z.open(name) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=e['sha256']:raise ValueError('Member checksum mismatch')
        if z.getinfo('manifest.json').file_size>1024**2:raise ValueError('Manifest too large')
        m=json.loads(z.read('manifest.json'))
        if m['run_id']!=index['run_id'] or m['cell_id']!=cell_id:raise ValueError('Run identity mismatch')
        reps=index['covered_replicates']
        if len(reps)>5 or len(reps)!=len(set(reps)) or any(type(r)!=int or not 0<=r<25 for r in reps):raise ValueError('Invalid archive coverage')
        for name in names:
            if name.startswith('shards/') and not any(name.startswith(f'shards/{r//5:02d}/rep_{r:02d}/') for r in reps):
                raise ValueError('Out-of-coverage raw data')
    return index


def choose_archive(paths,cell_id):
    paths=sorted(set(Path(p).resolve() for p in paths))
    if not paths:return None
    rows=[(p,inspect_zip(p,cell_id)) for p in paths]
    if len({i['run_id'] for p,i in rows})!=1:raise ValueError('Multiple runs: specify RESTORE_ZIP explicitly')
    return max(rows,key=lambda x:x[1]['created'])[0]


def restore(path,root,cell_id,sources):
    root=Path(root)
    if root.exists():raise FileExistsError('Never overwrite a live run')
    idx=inspect_zip(path,cell_id);root.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root.parent).free<sum(x['size'] for x in idx['entries'].values())+2*1024**3:
        raise RuntimeError('Insufficient restore space')
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='restore-') as td:
        stage=Path(td)/'run';stage.mkdir()
        with zipfile.ZipFile(path) as z:
            for name in idx['entries']:
                p=stage/allowed(name);p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,1024**2)
        check_run(stage,cell_id,sources)
        cat=catalog(stage)
        cat['archives'].append(dict(name=Path(path).name,sha256=digest(path),bytes=Path(path).stat().st_size,
            created=idx['created'],status=idx['status'],covered_replicates=idx['covered_replicates'],
            carry_hashes={k:v for k,v in idx['carry_hashes'].items() if int(k) in idx['covered_replicates']}))
        write(stage/'catalog.json',cat);remote=verify_carry(stage)
        os.replace(stage,root)
    return remote


class ActiveClock:
    def __init__(self,path,limit):
        self.path=Path(path);self.limit=limit;self.previous=read(path)['seconds'] if self.path.exists() else 0.;self.started=time.monotonic()
    def seconds(self):return self.previous+time.monotonic()-self.started
    def commit(self):write(self.path,dict(seconds=self.seconds(),at=stamp(),limitation='Abrupt VM loss can lose time since last committed chunk.'))
    def check(self):
        self.commit()
        if self.seconds()>=self.limit:raise TimeoutError('Cumulative active budget reached; not reset on resume')
