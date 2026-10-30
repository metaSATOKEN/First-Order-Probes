# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Resolve low-FPR rank-cap limitations using existing calibration and normal ranks."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
from analyze_pr0004_lowfpr import cal_arrays, ranks_from_cal, scores, load_arrays, ci, put, order, file_hash
import pr0004_spec as spec


def run(archive,out):
    out=Path(out);assert not (out/'tail_diagnostics.json').exists()
    md=json.loads((out/'metadata.json').read_text())
    with np.load(out/'replicate_metrics.npz',allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    nk={tuple(k):i for i,k in enumerate(md['null_keys'])};fk={tuple(k):i for i,k in enumerate(md['fault_keys'])}
    rows=[];taskmap={(t['replicate'],t['case'],t['role']):t for t in spec.tasks()}
    with zipfile.ZipFile(archive) as z:
        index=json.loads(z.read('ARCHIVE_INDEX.json'))['entries']
        def read(p):
            b=z.read(p);assert hashlib.sha256(b).hexdigest()==index[p]['sha256'];return b
        def arr(p):return load_arrays(read(p))
        def js(p):return json.loads(read(p))
        for rep in range(50):
            done=js(f'analysis/repeats/{rep:02d}/done.json');pre=f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
            models={}
            for group,c in spec.conditions().items():
                if c['kind']!='normal':continue
                model=js(pre+f'models/{group}/model.json');reference=arr(pre+f'models/{group}/reference_ranks.npz')
                task=taskmap[rep,group,'cal']
                values=np.concatenate([arr('chunks/'+task['id']+f'/{st:09d}.npz')['values'] for st in range(0,10000,4096)])
                models[group]={}
                for p in (['B'] if group.startswith('B_') else ['F','C','R']):
                    m,o=(model,reference) if p in ('B','R') else models[spec.baseline(group)]['B'][:2]
                    cc=models[spec.baseline(group)]['B'][2] if p=='F' else cal_arrays(ranks_from_cal(m,o,values))
                    models[group][p]=(m,o,cc)
                    theoretical=scores({'J':np.full((1,2),9999),'LF':np.full((1,2),9999)},cc)[0]
                    assert m['disabled']==[], 'This cap calculation requires both active branches'
                    saved=arr(pre+f'decisions/{group}_{p}.npz');u=scores(saved,cc)
                    assert np.all(u<=theoretical)
                    caprank=np.column_stack((saved['J'][:,0],saved['J'][:,1],saved['J'].max(1),saved['J'].max(1),saved['LF'].max(1)))
                    ni=nk[group,p]
                    for j,method in enumerate(md['methods']):
                        for ai,beta in enumerate(md['alphas']):
                            k=int(data['thresholds'][rep,ni,j,md['anchor_indices'][ai]])
                            disabled=k>int(theoretical[j]);nominal_disabled=order(beta,method=='B')>int(theoretical[j])
                            if disabled:
                                assert data['actual_fpr'][rep,ni,j,md['anchor_indices'][ai]]==0
                                for (case,pol),fi in fk.items():
                                    if pol==p and md['conditions'][case]['group']==group:
                                        assert data['power'][rep,fi,j,md['anchor_indices'][ai]]==0
                            rows.append(dict(replicate=rep,group=group,policy=p,method=method,beta=beta,
                                rank_cap_fraction=float((caprank[:,j]==9999).mean()),
                                calibrated_cap_fraction=float((u[:,j]==theoretical[j]).mean()),
                                maximum_possible_score=int(theoretical[j]),selected_threshold=k,
                                always_negative=disabled,nominal_always_negative=nominal_disabled))
            if (rep+1)%10==0:print(f'Tail diagnostics {rep+1}/50',flush=True)
    summary=[]
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[r['group'],r['policy'],r['method'],r['beta']].append(r)
    for key,rr in groups.items():
        assert len(rr)==50
        summary.append(dict(group=key[0],policy=key[1],method=key[2],beta=key[3],
            always_negative_replicates=sum(r['always_negative'] for r in rr),
            nominal_always_negative_replicates=sum(r['nominal_always_negative'] for r in rr),
            rank_cap_fraction=ci([r['rank_cap_fraction'] for r in rr]),
            calibrated_cap_fraction=ci([r['calibrated_cap_fraction'] for r in rr])))
    put(out/'tail_replicates.json',rows)
    put(out/'tail_diagnostics.json',dict(status='PASS',summary=summary,rows=len(rows),
        purpose='Post-hoc explanation of attainability, no rule changes',source_sha256=file_hash(Path(__file__))))


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
