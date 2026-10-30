# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""A01 bounded, non-cumulative transport; immutable evidence and delta recovery."""
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
from pr0006_io import ActiveClock

EXPERIMENT='mixture_recovery_pr0012_v1'
ZIP_LIMIT=1_000_000_000
PREPACK_LIMIT=800_000_000
TARGET_BYTES=600_000_000
CACHE_LIMIT=16*1024**3
MEMBERS_LIMIT=40000
ROOT_FILES={'manifest.json','specification.json','state.json','active_time.json','catalog.json',
            'summary.json','checkpoint_inventory.json','delivery_status.json'}
ROOT_DIRS={'sources','carry','sessions','shards','analysis'}
MUTABLE={'state.json','active_time.json','catalog.json','summary.json','checkpoint_inventory.json','delivery_status.json'}


def allowed(name):
    p=safe_name(name)
    if len(p.parts)==1 and name in ROOT_FILES:return p
    if len(p.parts)>1 and p.parts[0] in ROOT_DIRS:return p
    raise ValueError('Unexpected evidence path: '+name)


def new_run(root,cell_id,sources):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write(root/'manifest.json',dict(schema=1,experiment=EXPERIMENT,cell_id=cell_id,run_id=uuid.uuid4().hex[:12],
        created=stamp(),sources={n:sha(b.encode()) for n,b in sources.items()},
        card_sha256=sha(sources['PR-0012.md'].encode()),development_sha256=sha(sources['PR0012-development.json'].encode())))
    for name,body in sources.items():
        if Path(name).name!=name:raise ValueError('Source capsule name must be flat')
        atomic(root/'sources'/name,body.encode())
    write(root/'state.json',dict(completed={},status='NEW'))
    write(root/'catalog.json',dict(archives=[]))


def check_run(root,cell_id,sources):
    root=Path(root);m=read(root/'manifest.json');expected={n:sha(b.encode()) for n,b in sources.items()}
    if m.get('experiment')!=EXPERIMENT or m.get('cell_id')!=cell_id or m['sources']!=expected:
        raise ValueError('Run/source capsule mismatch')
    if (m['card_sha256']!=expected['PR-0012.md'] or m['development_sha256']!=expected['PR0012-development.json']):
        raise ValueError('Research-card hash mismatch')
    for n,h in expected.items():
        if digest(root/'sources'/n)!=h:raise ValueError('Saved source mismatch: '+n)
    return m


def rep_dir(root,rep):return Path(root)/'shards'/f'{rep//5:02d}'/f'rep_{rep:02d}'
def catalog(root):return read(Path(root)/'catalog.json')


def archived_hashes(root):
    out={}
    for a in catalog(root)['archives']:
        for n,h in a.get('evidence_hashes',{}).items():
            if n in out and out[n]!=h:raise ValueError('Conflicting archive receipts')
            out[n]=h
    return out


def verify_carry(root,records_per_rep=479984):
    root=Path(root);state=read(root/'state.json');remote=[];receipts=archived_hashes(root)
    ids=sorted(int(k) for k in state['completed'])
    if ids!=list(range(len(ids))) or len(ids)>50:raise ValueError('Completed replicates must be a contiguous prefix')
    if state['status']=='COMPLETE' and len(ids)!=50:raise ValueError('COMPLETE requires all 50 replicates')
    for key,record in state['completed'].items():
        rep=int(key)
        if key!=f'{rep:02d}':raise ValueError('Noncanonical replicate ID')
        meta=root/'carry'/f'rep_{rep:02d}.json';data=meta.with_suffix('.npz')
        if digest(meta)!=record['metadata_sha256']:raise ValueError('Carry metadata mismatch')
        d=read(meta)
        if d['replicate']!=rep or d['records']!=records_per_rep or digest(data)!=d['statistics_sha256']:
            raise ValueError('Carry identity/statistics mismatch')
        folder=rep_dir(root,rep);missing=False
        found={str(p.relative_to(folder)) for p in evidence(folder)}
        if found-set(d['files']):raise ValueError('Unknown completed evidence')
        for name,h in d['files'].items():
            p=folder/safe_name(name)
            if p.exists():
                if digest(p)!=h:raise ValueError('Completed evidence changed')
            elif receipts.get(str(p.relative_to(root)))==h:missing=True
            else:raise ValueError('Missing completed evidence without matching archive receipt')
        if missing:remote.append(rep)
    inv=root/'checkpoint_inventory.json'
    if inv.exists():
        for name,h in read(inv)['incomplete_files'].items():
            p=root/allowed(name)
            # No receipt-only bypass for an incomplete replicate: upload its delta dependencies.
            if not p.exists():raise ValueError('Missing ongoing delta dependency; restore ALL related PARTIAL ZIPs: '+name)
            if digest(p)!=h:raise ValueError('Incomplete checkpoint evidence changed')
    return remote


def update_inventory(root):
    root=Path(root);done=read(root/'state.json')['completed'];files={}
    for p in evidence(root/'shards'):
        rel=p.relative_to(root);rep=int(rel.parts[2].removeprefix('rep_'))
        if f'{rep:02d}' not in done:files[str(rel)]=digest(p)
    write(root/'checkpoint_inventory.json',dict(incomplete_files=files,created=stamp()))


def select_paths(root,reps,delta=False):
    root=Path(root);out=[];prior=archived_hashes(root) if delta else {}
    for p in evidence(root):
        rel=p.relative_to(root);allowed(str(rel))
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Symlink evidence rejected')
        if rel.parts[0]=='shards':
            if not any(p.is_relative_to(rep_dir(root,r)) for r in reps):continue
            if prior.get(str(rel))==digest(p):continue
        out.append(p)
    return out


def receipt(path,index):
    return dict(name=Path(path).name,sha256=digest(path),bytes=Path(path).stat().st_size,
        created=index['created'],sequence=index['sequence'],status=index['status'],
        covered_replicates=index['covered_replicates'],requested_replicates=index['requested_replicates'],
        evidence_hashes={n:e['sha256'] for n,e in index['entries'].items() if n.startswith('shards/')},
        target_exceeded=Path(path).stat().st_size>=TARGET_BYTES,zip_verified=True,
        browser_received=False,save_requested=False,user_disk_confirmed=False)


def snapshot_one(root,destination,status,reps,paths):
    root,destination=Path(root),Path(destination);m=read(root/'manifest.json')
    raw=sum(p.stat().st_size for p in paths)
    if raw>PREPACK_LIMIT or len(paths)+1>MEMBERS_LIMIT:raise OverflowError('Prepack budget exceeded')
    destination.mkdir(parents=True,exist_ok=True)
    if sum(p.stat().st_size for p in destination.glob('*.zip*'))+raw+16*1024**2>CACHE_LIMIT:
        raise TimeoutError('ZIP cache budget reached; evidence retained')
    entries={str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=digest(p)) for p in paths}
    covered=[r for r in reps if any(p.is_relative_to(rep_dir(root,r)) for p in paths)]
    index=dict(schema=1,experiment=EXPERIMENT,status=status,cell_id=m['cell_id'],run_id=m['run_id'],
        card_sha256=m['card_sha256'],development_sha256=m['development_sha256'],created=stamp(),
        sequence=len(catalog(root)['archives']),covered_replicates=covered,requested_replicates=list(reps),
        entries=entries,uncompressed_bytes=raw)
    name=EXPERIMENT+'_'+m['run_id']+'_'+status+'_'+uuid.uuid4().hex[:8]+'.zip'
    target=destination/name;tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in paths:z.write(p,str(p.relative_to(root)))
        z.writestr('ARCHIVE_INDEX.json',json.dumps(index,sort_keys=True))
    if tmp.stat().st_size>=ZIP_LIMIT:
        # This is an unpublished packing scratch file, never source evidence.
        tmp.unlink();raise OverflowError('Final ZIP >=1GB: split without changing scientific IDs')
    os.replace(tmp,target);inspect_zip(target,m['cell_id'])
    cat=catalog(root);cat['archives'].append(receipt(target,index));write(root/'catalog.json',cat)
    return target


def snapshot(root,destination,status,reps):
    """Try a whole five-replicate shard, then replicate, then committed-file boundaries."""
    root=Path(root);reps=sorted(set(reps))
    if len(reps)>5 or len({r//5 for r in reps})>1:raise ValueError('Cross-shard snapshot forbidden')
    update_inventory(root)
    delta=status in ('PARTIAL','TECHNICAL_FAILURE');paths=select_paths(root,reps,delta)
    raw=[p for p in paths if p.is_relative_to(root/'shards')]
    def pack(data,label):
        # Refresh metadata so the last part carries receipts for preceding parts.
        base=[p for p in select_paths(root,[],False)]
        try:return [snapshot_one(root,destination,label,reps,base+data)]
        except OverflowError:
            if len(data)<2:raise
            rs=sorted({int(p.relative_to(root).parts[2][4:]) for p in data})
            if len(rs)>1:
                mid=len(rs)//2;split=[p for p in data if int(p.relative_to(root).parts[2][4:]) in rs[:mid]]
                other=[p for p in data if int(p.relative_to(root).parts[2][4:]) not in rs[:mid]]
            else:split,other=data[:len(data)//2],data[len(data)//2:]
            return pack(split,label+'_A')+pack(other,label+'_B')
    return pack(raw,status)


def inspect_zip(path,cell_id,deadline=None):
    if Path(path).stat().st_size>=ZIP_LIMIT:raise ValueError('ZIP >= decimal 1 GB')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>MEMBERS_LIMIT:raise ValueError('Duplicate/excess archive members')
        if sum(i.file_size for i in infos)>PREPACK_LIMIT+16*1024**2:raise ValueError('Expanded archive too large')
        for i in infos:
            safe_name(i.filename)
            if i.is_dir() or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe ZIP member type')
            if i.filename!='ARCHIVE_INDEX.json':allowed(i.filename)
        if z.getinfo('ARCHIVE_INDEX.json').file_size>16*1024**2:raise ValueError('Index too large')
        idx=json.loads(z.read('ARCHIVE_INDEX.json'))
        if idx.get('schema')!=1 or idx.get('experiment')!=EXPERIMENT or idx.get('cell_id')!=cell_id:
            raise ValueError('Wrong experiment/cell')
        if set(names)!=set(idx['entries'])|{'ARCHIVE_INDEX.json'}:raise ValueError('Archive inventory mismatch')
        for name,e in idx['entries'].items():
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Restore verification exceeded 30 minutes')
            if z.getinfo(name).file_size!=e['size']:raise ValueError('ZIP member size mismatch')
            with z.open(name) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=e['sha256']:raise ValueError('ZIP member checksum mismatch')
        if z.getinfo('manifest.json').file_size>1024**2:raise ValueError('Manifest too large')
        m=json.loads(z.read('manifest.json'))
        for key in ('run_id','cell_id','card_sha256','development_sha256'):
            if idx[key]!=m[key]:raise ValueError('Manifest/index identity mismatch')
        reps=idx['covered_replicates']
        if len(reps)>5 or len(set(reps))!=len(reps) or any(type(r)!=int or not 0<=r<50 for r in reps):raise ValueError('Invalid coverage')
        if len({r//5 for r in reps})>1:raise ValueError('Cross-shard archive')
        for name in names:
            if name.startswith('shards/') and not any(name.startswith(f'shards/{r//5:02d}/rep_{r:02d}/') for r in reps):
                raise ValueError('Out-of-coverage data')
    return idx


def restore_many(paths,root,cell_id,sources):
    """Stream archives one at a time; hash-deduplicate immutable members, never execute sources."""
    root=Path(root)
    if root.exists():raise FileExistsError('Never overwrite a live run')
    paths=sorted(set(Path(p).resolve() for p in paths))
    deadline=time.monotonic()+1800
    rows=[(p,inspect_zip(p,cell_id,deadline)) for p in paths]
    if not rows or len({i['run_id'] for p,i in rows})!=1:raise ValueError('Select archives from exactly one run')
    rows.sort(key=lambda x:(x[1]['sequence'],x[1]['created']))
    root.parent.mkdir(parents=True,exist_ok=True)
    need=sum(e['size'] for p,i in rows for e in i['entries'].values())
    if shutil.disk_usage(root.parent).free<need+2*1024**3:raise RuntimeError('Insufficient restore space')
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='restore-') as td:
        stage=Path(td)/'run';stage.mkdir();seen={};receipts={}
        for path,idx in rows:
            with zipfile.ZipFile(path) as z:
                for name,e in idx['entries'].items():
                    p=stage/allowed(name);mutable=name in MUTABLE or name.startswith('sessions/')
                    if name in seen and not mutable:
                        if seen[name]!=e['sha256']:raise ValueError('Conflicting duplicate evidence: '+name)
                        continue
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,p.open('wb') as dst:
                        while True:
                            if time.monotonic()>deadline:raise TimeoutError('Restore exceeded 30 minutes')
                            chunk=src.read(1024**2)
                            if not chunk:break
                            dst.write(chunk)
                    seen[name]=e['sha256']
                for a in catalog(stage)['archives']:
                    if a['name'] in receipts and receipts[a['name']]['sha256']!=a['sha256']:raise ValueError('Conflicting receipt')
                    receipts[a['name']]=a
            receipts[path.name]=receipt(path,idx)
        write(stage/'catalog.json',dict(archives=sorted(receipts.values(),key=lambda a:(a['sequence'],a['created']))))
        check_run(stage,cell_id,sources);remote=verify_carry(stage)
        os.replace(stage,root)
    return remote
