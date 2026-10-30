# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Independent output-table and preserved-evidence checks for PR-0012."""
from pathlib import Path
import hashlib,json,re,copy
import numpy as np
from summarize_pr0012_confirmation import OUT,ROOT,G,C,E,N,CON,verify_primary

def read(name):return json.loads((OUT/name).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def close(value,expected):
    if expected is None:assert value=='NA';return
    # Only display rounding is loose. Scientific integer counts remain exact.
    assert abs(float(value.replace('−','-').replace('%',''))-100*expected)<=.000051,(value,expected)
def rows(name,first):
    return [[s.strip() for s in line.strip().strip('|').split('|')] for line in (OUT/name).read_text().splitlines()
            if line.startswith('| ') and line.split('|')[1].strip() in first]

def main():
    a=dict(np.load(OUT/'outer_arrays.npz'));p=read('primary13.json');checks={}
    checks['primary_classifications']=verify_primary(a,p)
    bad=copy.deepcopy(p);bad[-1]['classification']='SUPPORTED'
    try:verify_primary(a,bad)
    except AssertionError:checks['false_all_supported_negative_control']='REJECTED'
    else:raise AssertionError('False R05 support accepted')
    ps=read('policy_summary.json');pairs=read('paired_summary.json');cal=read('calibration_summary.json');lf=read('lf_summary.json')
    # Validate all summary values directly against integer arrays.
    for r in ps:
        g,c,e,ai,pi=G.index(r['group']),C.index(r['candidate']),E.index(r['evaluation']),(.01,.001).index(r['alpha']),int(r['policy'][1])
        h=a['policy'][:,g,c,e,ai,pi];tot=h.sum(0);den=50*N[e]
        for k,v in [('alarm',tot[0]/den),('quiet',tot[1]/den),('defer',tot[2]/den),('invalid',tot[3]/den),('coverage',tot[:2].sum()/den)]:
            assert abs(r[k]['mean']-v)<1e-14
        v=tot[0]/tot[:2].sum() if tot[:2].sum() else None
        assert r['provided_alarm']['mean'] is None if v is None else abs(r['provided_alarm']['mean']-v)<1e-14
    for r in cal:
        g,e,ai=G.index(r['group']),E.index(r['evaluation']),(.01,.001).index(r['alpha'])
        cs=CON if r['method']=='W0' else ['F','clean','C'];k='wcounts' if r['method']=='W0' else 'lcounts'
        h=a[k][:,g,e,cs.index(r['context']),ai].sum(0)/(50*N[e])
        assert abs(r['alarm']['mean']-sum(h[1:8]))<1e-14
        for j,bit in enumerate((1,2,4)):
            assert abs(r['branches'][j]['mean']-sum(h[b] for b in range(8) if b&bit))<1e-14
        assert abs(r['P_only']['mean']-h[4])<1e-14
    pr=rows('ALL_POLICIES.md',G);assert len(pr)==1152
    for row,r in zip(pr[:576],ps):
        assert row[:5]==[r['group'],r['candidate'],str(r['alpha']),r['policy'],r['evaluation']]
        for v,k in zip(row[5:],('alarm','coverage','provided_alarm','quiet','defer','invalid')):close(v,r[k]['mean'])
    for row,r in zip(pr[576:],pairs):
        assert row[:5]==[r['group'],r['candidate'],str(r['alpha']),r['pair'],r['evaluation']]
        for v,k in zip(row[5:],('gain','loss','net')):close(v,r[k]['mean'])
    cr=rows('ALL_CALIBRATIONS.md',G);assert len(cr)==448
    for row,r in zip(cr[:352],cal):
        assert row[:5]==[r['group'],r['evaluation'],str(r['alpha']),r['method'],r['context']]
        for v,x in zip(row[5:10],[r['alarm']['mean']]+[b['mean'] for b in r['branches']]+[r['P_only']['mean']]):close(v,x)
        assert int(row[10])==r['invalid']
    for row,r in zip(cr[352:],lf):
        assert row[:4]==[r['group'],r['evaluation'],str(r['alpha']),r['context']]
        for v,k in zip(row[4:7],('W0_only','LF0_only','net')):close(v,r[k]['mean'])
        assert int(row[7])==r['inactive']
    gr=rows('ALL_GATES_AND_SUPPORT.md',G);assert len(gr)==184
    for i,row in enumerate(gr[:24]):
        g,c=divmod(i,6);assert row[:2]==[G[g],C[c]]
        assert list(map(int,row[2:]))==(a['gates'][:,g,c]==1).sum(0).tolist()
    for row,r in zip(gr[24:168],read('selection_summary.json')):
        assert row[:4]==[r['group'],r['candidate'],str(r['alpha']),r['policy']]
        assert list(map(int,row[4:]))==[r['counts'][k] for k in ('F','U','C','DEFER','INVALID')]
    for row,r in zip(gr[168:],read('support_summary.json')):
        assert row[:3]==[r['group'],r['context'],str(r['alpha'])]
        assert json.loads(row[3])==r['V'] and json.loads(row[4])==r['E'] and json.loads(row[5])==r['cross']
        close(row[6],r['E_alarm']['mean'])
    checks['appendix_rows_checked']=len(pr)+len(cr)+len(gr)
    main='PR0012_CONFIRMATION_REPORT_JA.md';text=(OUT/main).read_text()
    mr=rows(main,[r['id'] for r in p]);assert len(mr)==13
    for row,r in zip(mr,p):
        assert row[0]==r['id'];close(row[2].replace('+',''),r['interval']['mean'])
        nums=re.findall(r'[−-]?\d+\.\d+',row[3]);assert len(nums)==2
        for n,k in zip(nums,('lower','upper')):close(n,r['interval'][k])
        assert ('未解決' in row[4])==(r['classification']=='UNRESOLVED')
    # Main recovery table: independent index selection, no implicit row offsets.
    pol=rows(main,['P0','P1','P2']);assert len(pol)==3
    for row in pol:
        pi=int(row[0][1]);h=a['policy'][:,1,5,:,0,pi].sum(0)
        vals=[h[0,:2].sum()/1e6,h[0,0]/1e6,h[0,0]/h[0,:2].sum() if h[0,:2].sum() else None,
              h[1,0]/250000,h[2,0]/250000,h[3,0]/250000,h[0,2]/1e6]
        for v,x in zip(row[1:],vals):close(v,x)
    # Main P-addition and gain/loss tables keyed by unique descriptions / row lengths.
    more=rows(main,['正常誤警報 %','負ゲイン検出 %','正ゲイン検出 %','私的入力検出 %']);assert len(more)==4
    for e,row in enumerate(more):
        h=a['wcounts'][:,2,e,:,0].sum(0);v1=h[5,1:8].sum()/(50*N[e]);v2=h[6,1:8].sum()/(50*N[e])
        for v,x in zip(row[1:],(v1,v2,v2-v1)):close(v,x)
    more=rows(main,['正常','負ゲイン','正ゲイン','私的入力']);assert len(more)==4
    for e,row in enumerate(more):
        h=a['pairs'][:,1,5,e,0,1].sum(0);gain=h[0,1:].sum()/(50*N[e]);loss=h[1:,0].sum()/(50*N[e])
        for v,x in zip(row[1:],(gain,loss,gain-loss)):close(v,x)
    checks['main_numeric_table_rows_checked']=24
    images=re.findall(r'!\[[^]]*\]\(([^)]+)\)',text);assert len(images)==8
    assert all(Path(f).is_file() for f in images)
    checks['figures']={Path(f).name:sha(Path(f).read_bytes()) for f in images}
    # Historical evidence is immutable, while corpus/ledger have append-only rules.
    start=read('start_snapshot.json');mutable={'audits/fp_corpus.md','audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md'}
    for path,h in start['anchors'].items():
        if path not in mutable:assert sha((ROOT/path).read_bytes())==h,path
    for path,key in [('audits/ledger_2026-10.md','ledger'),('audits/fp_corpus.md','corpus')]:
        old=start[key];assert sha((ROOT/path).read_bytes()[:old['bytes']])==old['sha256']
    checks['historical_anchors_unchanged']=len(start['anchors'])-len(set(start['anchors'])&mutable)
    # Source archives are unchanged since the full CRC / member audit.
    for r in read('receipt.json')['archives']:
        with (ROOT/'pr0012'/r['file']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    checks['source_archives_unchanged']=11
    checks.update(status='PASS_DELIVERY',independent_reviewer=False,scientific_generation=0,manuscript_edits=0)
    (OUT/'delivery_validation.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(checks,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
