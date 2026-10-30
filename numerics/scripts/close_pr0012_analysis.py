# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Check append-only navigation and freeze the PR-0012 analysis delivery."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def js(p):return json.loads(p.read_text())
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def main():
    start=js(OUT/'start_snapshot.json');delivery=js(OUT/'delivery_validation.json')
    assert delivery['status']=='PASS_DELIVERY'
    nav=['audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md']
    prior=js(ROOT/'audits/verification/2026-10-18_PR-0012_fix1_closeout.json')
    for path in nav:
        b=(ROOT/path).read_text();old,n=re.subn(r'\n<!-- PR0012-RESULTS-START -->\n.*?<!-- PR0012-RESULTS-END -->\n','',b,flags=re.S)
        assert n==1 and sha(old.encode())==prior['current_navigation'][path],path
    append={}
    for key,path in [('ledger','audits/ledger_2026-10.md'),('corpus','audits/fp_corpus.md')]:
        b=(ROOT/path).read_bytes();s=start[key]
        assert sha(b[:s['bytes']])==s['sha256']
        append[key]=dict(bytes=len(b),sha256=sha(b),preserved_prefix=dict(bytes=s['bytes'],sha256=s['sha256']))
    mutable=set(nav+['audits/fp_corpus.md'])
    protected={p:h for p,h in start['anchors'].items() if p not in mutable}
    for p,h in protected.items():assert sha((ROOT/p).read_bytes())==h,p
    visual=dict(status='EIGHT_FIGURES_DISPLAYED_AND_REVIEWED',reviewer='same agent',
       checks=['titles/conditions','axis units','legends','readable labels','R05 unresolved visible'],
       notes='Figures 1-8 were displayed through view_image during this work unit. Small intervals in figure 1 are also tabulated.',
       figures=delivery['figures'])
    for p,h in visual['figures'].items():assert sha((OUT/'figures'/p).read_bytes())==h
    dump(OUT/'visual_inspection.json',visual)
    new_sources=[ROOT/'numerics/scripts'/n for n in ['analyze_pr0012_return.py','summarize_pr0012_confirmation.py','verify_pr0012_delivery.py','close_pr0012_analysis.py']]
    docs=[ROOT/'audits/verification'/n for n in ['PR0012_ANALYSIS_SCOPE.md','2026-10-18_PR0012_early_summary_correction.md','2026-10-18_PR0012_confirmation_analysis.md']]
    outputs={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [*new_sources,*docs,*OUT.rglob('*')]
             if p.is_file() and p.name!='closeout_validation.json'}
    main=OUT/'PR0012_CONFIRMATION_REPORT_JA.md'
    for link in re.findall(r'\]\(([^)]+)\)',main.read_text()):
        q=Path(link) if link.startswith('/') else main.parent/link
        assert q.exists(),link
    result=dict(status='PASS_CLOSEOUT',protected_sources=protected,outputs=outputs,
      current_navigation={p:sha((ROOT/p).read_bytes()) for p in nav},**append,
      archives={r['file']:r['sha256'] for r in js(OUT/'receipt.json')['archives']},
      primary=delivery['primary_classifications'],
      tests=dict(full_saved_reconstruction=True,second_path_primary_and_counts=True,delivery_tables=True,visual_review=True),
      scientific_records_generated_locally=0,manuscript_edits=0,independent_reviewer=False,
      gaps=['waveforms not saved or regenerated','not a physical experiment','no VM-loss restore test',
            'U passage is not V certification','clean C/V supplied externally','R05 unresolved'])
    dump(OUT/'closeout_validation.json',result)
    print('PASS_CLOSEOUT',len(protected),'historical anchors;',len(outputs),'outputs; append-only ledger/corpus and 3 additive navigation blocks')
if __name__=='__main__':main()
