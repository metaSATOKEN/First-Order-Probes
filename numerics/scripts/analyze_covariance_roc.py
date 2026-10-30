# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read saved scores; never generate new probe records or execute archived code.

Produces descriptive empirical ROC, a fixed calibration/holdout split, and
paired nonparametric bootstrap intervals at 1% FPR. Manuscripts are untouched.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import zipfile

import numpy as np

VERSION = 'roc-postanalysis-1.0.0'
SEED = 2026101702
BOOTSTRAPS = 1000
METHODS = ('moment', 'CR')
ALPHAS = (.001, .005, .01, .02, .05)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = ROOT / 'main_2ae37613a10478c8_COMPLETE_20261016T160008Z.zip'
DEFAULT_OUT = ROOT / 'numerics/analysis/main_2ae37613a10478c8'


def rng_for(label):
    words = np.frombuffer(hashlib.sha256(label.encode()).digest(), dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([SEED, *map(int, words)]))


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def dump_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold(values, alpha):
    """Strict score > q. No normal-score ties: exactly alpha*n false positives."""
    n = len(values)
    k = int(round(alpha*n))
    if abs(k-alpha*n)>1e-9 or not 0<=k<n:
        raise ValueError('FPR must be representable on the normal-sample grid')
    return float(np.partition(values, n-k-1)[n-k-1])


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., center-half), min(1., center+half)


def auc_rank(normal, alternative):
    ordered = np.sort(normal)
    left = np.searchsorted(ordered, alternative, side='left')
    right = np.searchsorted(ordered, alternative, side='right')
    return float(np.mean((left+right)/2)/len(ordered))


def normalized_pauc(normal, alternative, limit=.05):
    # Exact empirical area for tie-free normal scores, not a sparse-grid integral.
    k = int(round(limit*len(normal)))
    cutoffs = np.sort(normal)[-k:]
    alt = np.sort(alternative)
    tp = (len(alt)-np.searchsorted(alt, cutoffs, side='right'))/len(alt)
    assert not np.any(np.isin(cutoffs, alt)), 'Cross-class ties need explicit ROC handling'
    return float(tp.mean())


def deterministic_checks():
    n = np.array([0., 1., 2., 3.])
    a = np.array([.5, 2.5, 4.])
    assert np.mean(n>threshold(n,.25))==.25
    brute = np.mean([float(x>y)+.5*float(x==y) for x in a for y in n])
    assert abs(auc_rank(n,a)-brute)<1e-14
    assert abs(normalized_pauc(n,a,.5)-.5)<1e-14
    assert auc_rank(n,np.array([4.,5.]))==1
    assert auc_rank(n,np.array([-2.,-1.]))==0
    assert auc_rank(n,n)==.5


class Archive:
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        assert len(names)==len(set(names)), 'Duplicate archive entries'
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        self.config = self.read_json('config.json')
        self.status = self.read_json('run_status.json')
        self.design = self.read_json('design.json')
        self.tables = {PurePosixPath(n).stem:list(csv.DictReader(io.StringIO(self.z.read(n).decode())))
                       for n in names if n.startswith('tables/') and n.endswith('.csv')}
        self.chunks = collections.defaultdict(list)
        cfg_hash = hashlib.sha256(json.dumps(self.config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert cfg_hash==self.status['config_hash']
        engine = self.z.read('runner_source.py').decode().split('# === ENGINE START ===',1)[1]
        assert hashlib.sha256(engine.encode()).hexdigest()==self.config['engine_sha256']
        for name in names:
            if name.startswith('chunks/') and name.endswith('.json'):
                m = self.read_json(name)
                npz = name[:-5]+'.npz'
                assert hashlib.sha256(self.z.read(npz)).hexdigest()==m['sha256'],name
                assert m['config_hash']==cfg_hash and name.split('/')[1]==m['task']
                self.chunks[m['task']].append((m['start'],m['count'],npz))
        expected = {}
        sizes = self.config['repetitions']
        for c in self.design:
            expected[c['reference_id']]=sizes['reference']
            expected[c['id']]=(sizes['endpoint'] if c['generator']=='endpoint' else
                               sizes['alternative'] if c['kind']=='alternative' else sizes['null'])
        assert set(expected)==set(self.chunks)
        for task, pieces in self.chunks.items():
            offset=0
            for start,count,_ in sorted(pieces):
                assert start==offset,(task,start,offset)
                offset+=count
            assert offset==expected[task]
        assert self.status['status']=='COMPLETE' and self.status['completed_cases']==len(self.design)==146
        self.audit=dict(cases=len(self.design),tasks=len(expected),records=sum(expected.values()),
                        chunks=sum(map(len,self.chunks.values())),chunk_sha256_verified=True,
                        contiguous_counts_verified=True,archived_engine_sha256_verified=True)

    def read_json(self,name):
        return json.loads(self.z.read(name))

    def scores(self,task):
        pieces=[]
        for _,count,name in sorted(self.chunks[task]):
            with np.load(io.BytesIO(self.z.read(name)),allow_pickle=False) as data:
                a=np.column_stack((np.abs(data['R_full']),data['CR']))
                assert a.shape==(count,2) and np.isfinite(a).all()
                pieces.append(a)
        return np.concatenate(pieces)


def bootstrap_thresholds(normal,label):
    rng=rng_for('normal-bootstrap/'+label)
    n=len(normal); rank=n-int(.01*n)-1
    output=np.empty((BOOTSTRAPS,2))
    for start in range(0,BOOTSTRAPS,32):
        count=min(32,BOOTSTRAPS-start)
        ix=rng.integers(0,n,size=(count,n))  # shared indices preserve method pairing
        for m in range(2):
            output[start:start+count,m]=np.partition(normal[ix,m],rank,axis=1)[:,rank]
    return output


def bootstrap_power(alt,thresholds,label):
    rng=rng_for('alternative-bootstrap/'+label)
    n=len(alt); powers=np.empty((BOOTSTRAPS,2))
    for start in range(0,BOOTSTRAPS,32):
        count=min(32,BOOTSTRAPS-start)
        ix=rng.integers(0,n,size=(count,n))
        for m in range(2):
            powers[start:start+count,m]=(alt[ix,m]>thresholds[start:start+count,m,None]).mean(axis=1)
    return np.quantile(powers,[.025,.975],axis=0),np.quantile(powers[:,0]-powers[:,1],[.025,.975])


def analyze(path,out):
    start=time.monotonic()
    deterministic_checks()
    out.mkdir(parents=True,exist_ok=True)
    (out/'tables').mkdir(exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    arc=Archive(path)
    cases=[c for c in arc.design if c['experiment']==3]
    normals={c['reference_id']:c for c in cases if c['kind']=='null'}
    alternatives=[c for c in cases if c['kind']=='alternative']
    assert len(normals)==9 and len(alternatives)==90
    cached={}; normal_rows=[]; split_rows=[]
    print('Archive verified. Preparing nine paired normal-score groups.',flush=True)
    for rid,c in normals.items():
        score=arc.scores(c['id']); n=len(score)
        assert n==20000
        for m in range(2):
            assert len(np.unique(score[:,m]))==n, 'Cannot claim exact FPR with ties'
        ix=rng_for('holdout-split/'+c['id']).permutation(n)
        cal,hold=score[ix[:n//2]],score[ix[n//2:]]
        th=np.array([threshold(cal[:,m],.01) for m in range(2)])
        boot=bootstrap_thresholds(score,c['id'])
        cached[rid]=(score,th,boot)
        for m,method in enumerate(METHODS):
            hits=int(np.sum(hold[:,m]>th[m])); lo,hi=wilson(hits,len(hold))
            split_rows.append(dict(null_case=c['id'],method=method,calibration_n=len(cal),holdout_n=len(hold),
                target_fpr=.01,threshold=th[m],calibration_fpr=float(np.mean(cal[:,m]>th[m])),
                holdout_rejections=hits,holdout_fpr=hits/len(hold),holdout_wilson_low=lo,holdout_wilson_high=hi))
            for alpha in ALPHAS:
                cut=threshold(score[:,m],alpha); rate=float(np.mean(score[:,m]>cut))
                assert rate==alpha
                normal_rows.append(dict(null_case=c['id'],method=method,normal_n=n,target_fpr=alpha,
                                        threshold=cut,empirical_fpr=rate))
    roc_rows=[]; equal_rows=[]; pair_rows=[]; auc_rows=[]; holdout_power=[]; plot={}
    for i,c in enumerate(alternatives):
        alt=arc.scores(c['id']); normal,split_th,boot=cached[c['reference_id']]
        assert len(alt)==5000
        ci,dci=bootstrap_power(alt,boot,c['id'])
        nullcase=normals[c['reference_id']]['id']; p=c['params']
        common=dict(case=c['id'],null_case=nullcase,group=c['group'],fault=p['fault'],strength=p['strength'],T=p['T'],
                    tau_i=p['tau_i'],tau_j=p['tau_j'],dt=p['dt'],kappa=p['kappa'],eta_readout=p['eta_readout'])
        plotted={}
        for m,method in enumerate(METHODS):
            obs=np.sort(alt[:,m]); norm=np.sort(normal[:,m])
            # Exact rank-based AUC and partial AUC, plus a display grid up to 10% FPR.
            auc_rows.append(dict(**common,method=method,normal_n=len(norm),alternative_n=len(obs),
                auc=auc_rank(norm,obs),normalized_pauc_0_to_005=normalized_pauc(norm,obs)))
            grid=np.arange(0,2001,10)/len(norm)
            thresholds=norm[len(norm)-(grid*len(norm)).round().astype(int)-1]
            power=(len(obs)-np.searchsorted(obs,thresholds,side='right'))/len(obs)
            assert np.all(np.diff(power)>=0)
            plotted[method]=dict(fpr=grid.tolist(),tpr=power.tolist())
            for fp,tp,cut in zip(grid,power,thresholds):
                roc_rows.append(dict(case=c['id'],null_case=nullcase,method=method,fpr=float(fp),tpr=float(tp),threshold=float(cut)))
            for alpha in ALPHAS:
                cut=threshold(norm,alpha); hits=int(np.sum(alt[:,m]>cut))
                equal_rows.append(dict(**common,method=method,normal_n=len(norm),alternative_n=len(obs),target_fpr=alpha,
                    empirical_fpr=float(np.mean(norm>cut)),threshold=cut,detections=hits,tpr=hits/len(obs),
                    bootstrap_low=float(ci[0,m]) if alpha==.01 else '',
                    bootstrap_high=float(ci[1,m]) if alpha==.01 else ''))
            hits=int(np.sum(alt[:,m]>split_th[m])); lo,hi=wilson(hits,len(alt))
            holdout_power.append(dict(**common,method=method,target_calibration_fpr=.01,threshold=split_th[m],
                alternative_n=len(alt),detections=hits,tpr=hits/len(alt),
                conditional_wilson_low=lo,conditional_wilson_high=hi))
        a,b=[float(np.mean(alt[:,m]>threshold(normal[:,m],.01))) for m in range(2)]
        pair_rows.append(dict(**common,target_fpr=.01,moment_tpr=a,CR_tpr=b,difference=a-b,
                              paired_bootstrap_low=float(dci[0]),paired_bootstrap_high=float(dci[1])))
        plot[c['id']]=dict(params=p,group=c['group'],roc=plotted)
        if (i+1)%15==0: print(f'Analyzed {i+1}/{len(alternatives)} saved conditions.',flush=True)
    for name,rows in [('normal_thresholds',normal_rows),('matched_fpr_power',equal_rows),
                      ('paired_power_difference',pair_rows),('roc_grid',roc_rows),('auc',auc_rows),
                      ('holdout_false_alarm',split_rows),('holdout_power',holdout_power)]:
        dump_csv(out/'tables'/f'{name}.csv',rows)
    summary={
        'version':VERSION,'analysis_seed':SEED,'bootstrap_replicates':BOOTSTRAPS,
        'archive':path.name,'archive_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
        'original_config':arc.config,'original_status':arc.status,'audit':arc.audit,
        'analysis_python':platform.python_version(),'analysis_numpy':np.__version__,
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Post-hoc reanalysis of stored scores only. No new physical or simulated records.',
        'normal_groups':len(normals),'alternative_cases':len(alternatives),
        'counts_by_group':dict(collections.Counter(c['group'] for c in alternatives)),
        'bootstrap_scope':'Paired records within each normal/alternative group; re-estimate 1% threshold each replicate. Pointwise percentile intervals, not simultaneous.',
        'split_scope':'Deterministic 10000 calibration / 10000 holdout normal records, shared between methods. Alternatives never set thresholds. Exploratory, not a new experiment.',
        'elapsed_seconds':time.monotonic()-start,
    }
    dump_json(out/'audit.json',summary)
    dump_json(out/'plot_data.json',dict(cases=plot,power=equal_rows,pairs=pair_rows))
    print(json.dumps(summary,indent=2),flush=True)


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((out/'plot_data.json').read_text())
    colors={'moment':'#1768AC','CR':'#D55E00'}
    labels={'moment':'Moment |R|','CR':'Cross-relation energy'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none'})
    def save(fig,name):
        for suffix in ('png','svg'):
            fig.savefig(out/'figures'/f'{name}.{suffix}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def roc(ax,cid,title):
        for method in METHODS:
            r=d['cases'][cid]['roc'][method]
            x=np.array(r['fpr']); y=np.array(r['tpr']); keep=x<=.05
            ax.plot(x[keep]*100,y[keep]*100,color=colors[method],label=labels[method],lw=2)
            ix=int(np.flatnonzero(np.isclose(x,.01))[0])
            ax.scatter([1],[100*y[ix]],color=colors[method],s=30,zorder=3)
        ax.axvline(1,color='#888888',ls=':',lw=1)
        ax.set(xlim=(0,5),ylim=(0,102),title=title,xlabel='Empirical false-positive rate (%)',ylabel='Detection rate (%)')
        ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.8),layout='constrained')
    for row,gain in enumerate((-.05,.05)):
        for col,T in enumerate((.512,2.048,8.192)):
            roc(axes[row,col],f'e3_gain_{gain:g}_T{T:g}',f'Gain {gain:+.0%}; T = {T:g} s')
    axes[0,0].legend(loc='upper left',fontsize=9)
    fig.suptitle('Gain-shift ROC across window lengths\nSaved-data empirical ROC; dots mark equal 1% FPR',fontsize=14)
    save(fig,'roc_gain_by_window')
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.6),layout='constrained')
    for ax,T in zip(axes,(.512,2.048,8.192)):
        for method in METHODS:
            rows=sorted([r for r in d['power'] if r['group']=='primary' and r['fault']=='gain'
                          and r['T']==T and r['method']==method and r['target_fpr']==.01],key=lambda r:r['strength'])
            x=np.array([r['strength']*100 for r in rows]); y=np.array([r['tpr']*100 for r in rows])
            lo=np.array([r['bootstrap_low']*100 for r in rows]); hi=np.array([r['bootstrap_high']*100 for r in rows])
            # Do not interpolate across the untested zero-error region.
            for sign in (-1,1):
                select=x*sign>0
                ax.plot(x[select],y[select],'o-',color=colors[method],
                        label=labels[method] if sign==-1 else None,lw=1.8,ms=4)
            ax.vlines(x,lo,hi,color=colors[method],lw=1.2)
        ax.axhline(1,color='#888888',ls=':',lw=1)
        ax.scatter([0],[1],marker='x',color='#555555',s=28,zorder=3)
        ax.set(xlim=(-10.5,10.5),ylim=(0,102),title=f'T = {T:g} s',xlabel='Signal gain error (%)',ylabel='Detection rate (%)')
        ax.set_xticks([-10,-5,-2,0,2,5,10])
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle('Gain sensitivity at equal empirical 1% FPR\nPointwise 95% paired bootstrap: normal-threshold and alternative uncertainty',fontsize=13)
    save(fig,'gain_power_equal_fpr')
    fig,axes=plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
    selected=[('e3_private_0.02_T2.048','Private-input variance ratio 0.02'),
              ('e3_blind_time_reverse_T2.048','Time reversal (structural control)'),
              ('e3_delay_-2_T2.048','Delay -2 samples (-4 ms)'),
              ('e3_delay_2_T2.048','Delay +2 samples (+4 ms)')]
    for ax,(cid,title) in zip(axes.flat,selected): roc(ax,cid,title)
    axes[0,0].legend(loc='center right',fontsize=9)
    fig.suptitle('Complementary sensitivities and blind spots; T = 2.048 s\nSaved-data empirical ROC; dots mark equal 1% FPR',fontsize=13)
    save(fig,'roc_complementary_cases')
    dump_json(out/'figure_environment.json',dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__))
    print('Rendered three figures as PNG and SVG.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=DEFAULT_ZIP)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--analyze-only',action='store_true')
    group.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if not args.plots_only: analyze(args.archive,args.out)
    if not args.analyze_only: plots(args.out)
