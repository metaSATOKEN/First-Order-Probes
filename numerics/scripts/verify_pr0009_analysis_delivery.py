# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Final saved-evidence inventory, chronology, report-table and preservation check."""
import io,json,re,hashlib
from pathlib import Path
import numpy as np
from analyze_pr0009_return import ROOT,OUT,Store,dump,sha
import pr0009_spec as spec

def main():
    s=Store()
    for seq,p,idx in s.indices:
        for n,e in idx['entries'].items():s.map[n]=p.name;s.hashes[n]=e['sha256']
    chronology=[];rss=0;gpu=0;times={};maxchunk=0
    for r in range(50):
        folder=f'shards/{r//5:02d}/rep_{r:02d}';expected=set();early=[];late=[]
        for task in spec.tasks(r):
            for start in range(0,task['count'],spec.CHUNK):
                p=folder+f"/chunks/{task['case']}/{task['role']}/{start:09d}.npz";expected.add(p)
                with np.load(io.BytesIO(s.read(p)),allow_pickle=False) as z:d=json.loads(str(z['metadata']))
                (late if task['role'].startswith('validation') or task['role'].endswith('_eval') else early).append(d['at'])
                rss=max(rss,d['memory']['rss_high_water_bytes']);gpu=max(gpu,d['memory']['gpu_reserved_bytes'])
                maxchunk=max(maxchunk,sum(d['timing'].values()))
                for k,v in d['timing'].items():times[k]=times.get(k,0)+v
        assert expected=={n for n in s.map if n.startswith(folder+'/chunks/')}
        assert max(early)<min(late)
        chronology.append(dict(replicate=r,latest_training_record=max(early),earliest_validation_or_evaluation_record=min(late),raw_chunks=len(expected)))
    dump(OUT/'chronology_and_resources.json',dict(chronology=chronology,all_calibration_inputs_before_V_E=True,
        raw_rss_high_water_bytes=rss,raw_gpu_reserved_max_bytes=gpu,raw_timing_sums=times,
        max_sum_of_chunk_timing_components=maxchunk,interpretation='Saved timestamps and resource observations; not independent clock instrumentation'))
    summary=json.loads((OUT/'report_summary.json').read_text());report=(OUT/'PR0009_TWO_AXIS_REPORT_JA.md').read_text()
    primary=[l for l in report.splitlines() if re.match(r'\| P\d\d \|',l)]
    assert len(primary)==18
    for l,p in zip(primary,summary['primary']):
        cells=[x.strip() for x in l.split('|')[1:-1]]
        assert cells[0]==p['id']
        assert cells[2:5]==[f"{100*p[k]:.3f}" for k in ('U','GMC','mean')]
        assert cells[5]==f"[{100*p['lower']:.3f}, {100*p['upper']:.3f}]"
    records=json.loads((OUT/'all_policy_summary.json').read_text())
    lines=[l for l in (OUT/'ALL_POLICY_RESULTS.md').read_text().splitlines() if any(l.startswith('| '+g+' |') for g in spec.groups())]
    assert len(lines)==len(records)==3528
    for l,r in zip(lines,records):
        cells=[x.strip() for x in l.split('|')[1:-1]]
        assert cells[0]==r['group'] and cells[2]==r['policy']
        assert cells[3:7]==[f"{100*r[k]['mean']:.3f}" for k in ('normal','GM05','GP05','P02')]
        assert cells[7:10]==[f"{100*(1-r[k]['mean']):.3f}" for k in ('GM05','GP05','P02')]
    links=0
    for p in OUT.glob('*.md'):
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            assert (p.parent/dest.split('#')[0]).resolve().exists(),(p,dest);links+=1
    for p in (OUT/'figures').glob('*.png'):
        from PIL import Image
        with Image.open(p) as im:im.verify()
    assert len(list((OUT/'figures').glob('*.png')))==8
    snapshot=json.loads((OUT/'start_snapshot.json').read_text());nav=['audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md','audits/ledger_2026-10.md']
    for p,h in snapshot['anchors'].items():
        if p not in nav:assert sha((ROOT/p).read_bytes())==h,p
    oldledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()[:snapshot['ledger_bytes']]
    assert sha(oldledger)==snapshot['anchors']['audits/ledger_2026-10.md']
    artifacts={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_validation.json'}
    sources={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),ROOT/'numerics/scripts/analyze_pr0009_return.py',ROOT/'numerics/scripts/report_pr0009_two_axis.py']}
    dump(OUT/'delivery_validation.json',dict(status='PASS_DELIVERY',primary_rows=18,policy_table_rows=3528,local_links=links,
        figures=8,chronology_replicates=50,protected_anchors_unchanged=len(snapshot['anchors'])-len(nav),ledger_prefix_preserved=True,
        sources=sources,outputs=artifacts,navigation={p:sha((ROOT/p).read_bytes()) for p in nav},
        independent_reviewer=False,scientific_records_generated=0,manuscript_edits=0,
        visual_check='Main agent inspected all eight figures; heatmap contrast improved before finalization'))
    print('PASS_DELIVERY',len(artifacts),'outputs',links,'links',rss,'RSS',gpu,'GPU')
if __name__=='__main__':main()
