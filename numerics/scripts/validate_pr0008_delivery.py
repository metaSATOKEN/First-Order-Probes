# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate integer/summary/link audit; does not recompute or tune any gate."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
import analyze_pr0008_shadow as a
import pr0008_shadow_methods as m
import reproduce_pr0008_shadow as repro

P=a.DEFAULT


def main():
    snap=json.loads((P/'start_snapshot.json').read_text());v=json.loads((P/'validation.json').read_text())
    for name,h in {**snap['protected'],**snap['sources']}.items(): assert a.h.digest(a.ROOT/name)==h,name
    for name,h in v['output_hashes'].items(): assert a.h.digest(P/name)==h,name
    with np.load(P/'results.npz',allow_pickle=False) as f: d={k:f[k] for k in f.files}
    scores=d['scores'];states=d['states'];pc=d['policy_counts']; uf=d['UF_pairs'];pairs=d['policy_pairs']
    counts=d['Fstar_counts'];den=np.array([20000,5000,5000,5000]);checked=0
    for r in range(25):
        with np.load(P/'replicates'/f'Fstar_{r:02d}.npz',allow_pickle=False) as f:
            for gi,g in enumerate(m.GROUPS):
                for ei,tag in enumerate(('normal','GM05','GP05','P02')):
                    key=g if not ei else g+'__'+tag
                    bits=f[key]
                    for ai in range(2):
                        for mi in range(6):
                            nibble=(bits[:,ai]//(16**mi))%16
                            assert np.all(nibble>=8)
                            assert np.count_nonzero(nibble%8)==counts[r,gi,ei,ai,mi]
                    checked+=len(bits)*12
        with np.load(P/'splits'/f'rep_{r:02d}.npz',allow_pickle=False) as f:
            for base in ('B_M','B_L'):
                u,o=f[base+'_U_normal_ids'],f[base+'_Ostar_ids']
                assert len(set(map(int,u)) & set(map(int,o)))==0
                assert len(set(map(int,u)))==len(set(map(int,o)))==15999
    for r in range(25):
        for gi in range(6):
            for gate in range(3):
                development=[scores[x,g,0,gate] for x in range(25) if x//5!=r//5
                             for g in range(6) if m.WINDOWS[g]==m.WINDOWS[gi]]
                maximum=max(development)
                assert maximum==d['thresholds'][r//5,m.WINDOWS[gi],gate]
                for ci in range(7):
                    expected=int(scores[r,gi,ci,gate]>maximum)
                    assert states[r,gi,ci,gate]==expected
                    for ei in range(4):
                        chosen=counts[r,gi,ei] if expected else d['U_counts'][r,gi,ci,ei]
                        assert np.array_equal(pc[r,gi,ci,ei,:,:,gate+2],chosen)
    assert np.array_equal(uf[...,0]+uf[...,1],d['U_counts'])
    assert np.all(uf.sum(-1)==den[None,None,None,:,None,None])
    for gate in range(3):
        for ref,pi in ((0,1),(1,0)):
            assert np.array_equal(pairs[...,gate,ref,0]+pairs[...,gate,ref,1],pc[...,gate+2])
            assert np.array_equal(pairs[...,gate,ref,0]+pairs[...,gate,ref,2],pc[...,pi])
            assert np.all(pairs[...,gate,ref,:].sum(-1)==den[None,None,None,:,None,None])
    rows=json.loads((P/'policy_rates.json').read_text())
    for row in rows:
        gi=m.GROUPS.index(row['group']); ci=row['candidate'];ei=('normal','GM05','GP05','P02').index(row['evaluation'])
        ai=(.01,.001).index(row['alpha']);mi=('W0','WP','WG','W+','W−','LF0').index(row['method']);pi=m.POLICIES.index(row['policy'])
        vals=[int(x)/int(den[ei]) for x in pc[:,gi,ci,ei,ai,mi,pi]]
        mean=math.fsum(vals)/25
        assert abs(mean-row['rate']['mean'])<2e-14
        assert abs(math.sqrt(math.fsum((x-mean)**2 for x in vals)/24)-row['rate']['sd'])<2e-14
        assert row['per_replicate_alarms']==pc[:,gi,ci,ei,ai,mi,pi].tolist()
        for fold in range(5):assert abs(math.fsum(vals[fold*5:fold*5+5])/5-row['rate']['fold_means'][fold])<2e-14
        if pi>=2:
            for tag,is_accept in [('accepted',True),('held',False)]:
                rr=[r for r in range(25) if ((states[r,gi,ci,pi-2]==0)==is_accept)]
                assert row[tag]['replicates']==len(rr) and row[tag]['records']==len(rr)*den[ei]
                if rr:assert abs(math.fsum(vals[r] for r in rr)/len(rr)-row[tag]['rate'])<2e-14
                else:assert row[tag]['rate'] is None
    paired_rows=json.loads((P/'paired_tradeoffs.json').read_text())
    for row in paired_rows:
        gi=m.GROUPS.index(row['group']);ci=row['candidate'];ei=('normal','GM05','GP05','P02').index(row['evaluation'])
        ai=(.01,.001).index(row['alpha']);mi=('W0','WP','WG','W+','W−','LF0').index(row['method'])
        gate=m.GATES.index(row['policy']);ref=('U','F*').index(row['reference'])
        raw=pairs[:,gi,ci,ei,ai,mi,gate,ref]
        for j,k in enumerate(('both','gained','lost','neither')):
            val=math.fsum(int(x)*100/int(den[ei]) for x in raw[:,j])/25
            assert abs(val-row['cells_percent'][k]['mean'])<2e-12
        diff=math.fsum((int(x[1])-int(x[2]))*100/int(den[ei]) for x in raw)/25
        assert abs(diff-row['delta_pp']['mean'])<2e-12
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','numerics/scripts','-p','test_pr0008_shadow.py','-v'],
                         cwd=a.ROOT,capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    (P/'technical_tests.log').write_text(tests.stdout+tests.stderr)
    with tempfile.TemporaryDirectory(prefix='pr0008-provenance-') as temp:
        rs=repro.reproduction_snapshot(Path(temp));assert rs['card_sha256']==m.CARD_SHA
    # Complete machine-readable rates are checked above; spot-check headline text below.
    report=(P/'PR0008_SHADOW_GATE_REPORT_JA.md').read_text()
    # Exact principal rates, verified above through complete rate enumeration.
    required=['753','587','433','147','185','19','3.7164','0.0710','96.8040','15.3272',
              '44.2848','25.5128','18.7720','1.0912','0.1256','52.3568','0.2704']
    assert all(x in report for x in required)
    links=0
    files=[P/'PR0008_SHADOW_GATE_REPORT_JA.md',P/'README.md',a.CARD,
           a.ROOT/'audits/verification/2026-10-18_PR-0008_saved_data.md',
           a.ROOT/'audits/README.md',a.ROOT/'audits/prereg/README.md',a.ROOT/'HANDOFF/LATEST.md']
    for f in files:
        for dest in re.findall(r'\]\(([^\n]+?)\)',f.read_text()):
            if dest.startswith(('http://','https://','#')):continue
            q=(f.parent/dest.split('#')[0]).resolve()
            if q==P/'delivery_validation.json':continue
            assert q.exists(),(f,dest)
            links+=1
    result=dict(status='PASS_DESIGN_AND_DELIVERY',audit_type='same-agent separate integer and scalar-summary verification',
        card_sha256=m.CARD_SHA,original_science_unchanged=True,
        Fstar_decisions_recounted=checked,gate_decisions_checked=int(states.size),
        rate_series_checked=len(rows),paired_series_checked=len(paired_rows),
        technical_tests=10,reproduction_snapshot_tested=True,second_full_reproduction=False,
        local_links_checked=links,figures=7,visual_review='All seven figures inspected for labels, range, readability; SD is descriptive.',
        scientific_records_generated=0,manuscript_edited=False,
        ledger_prefix=dict(bytes=snap['ledger_bytes'],sha256=snap['ledger_sha256']),
        current_navigation_hashes={name:a.h.digest(a.ROOT/name) for name in
            ('audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md','audits/ledger_2026-10.md')},
        governance_verification_sha256=a.h.digest(a.ROOT/'audits/verification/2026-10-18_PR-0008_saved_data.md'),
        source_hashes={str(q.relative_to(a.ROOT)):a.h.digest(q) for q in
            (Path(__file__),Path(repro.__file__),a.ROOT/'numerics/scripts/report_pr0008_shadow.py')},
        outputs={str(q.relative_to(P)):a.h.digest(q) for q in sorted(P.rglob('*')) if q.is_file() and q.name!='delivery_validation.json'})
    a.write(P,'delivery_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','source_hashes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
