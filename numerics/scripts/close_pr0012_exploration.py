# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Close the saved-data exploration; verify preservation, never rewrite old evidence."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_exploration_v2'
SRC=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
NAV=['audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md']

def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(path.read_bytes())
def load(path):return json.loads(path.read_text())

def main():
    target=OUT/'closeout_validation.json'
    assert not target.exists(),'Do not replace completed closeout'
    snap=load(OUT/'start_snapshot.json')
    protected={}
    for name,h in snap['anchors'].items():
        if name in NAV:continue
        assert sha(ROOT/name)==h,name
        protected[name]=h
    old_close=SRC/'closeout_validation.json'
    assert sha(old_close)==snap['previous_closeout_sha']
    protected[str(old_close.relative_to(ROOT))]=sha(old_close)
    navigation={}
    for name in NAV:
        text=(ROOT/name).read_text()
        pattern=r'<!-- PR0012-EXPLORATION-START -->\n.*?<!-- PR0012-EXPLORATION-END -->\n\n'
        restored,n=re.subn(pattern,'',text,flags=re.S)
        assert n==1 and digest(restored.encode())==snap['anchors'][name],name
        navigation[name]=sha(ROOT/name)
    prefixes={}
    for key,name in [('ledger','audits/ledger_2026-10.md'),('corpus','audits/fp_corpus.md')]:
        old=snap[key];data=(ROOT/name).read_bytes()
        assert len(data)>old['bytes'] and digest(data[:old['bytes']])==old['sha256']
        prefixes[key]={'bytes':len(data),'sha256':digest(data),'preserved_prefix':{'bytes':old['bytes'],'sha256':old['sha256']}}
    delivery=load(OUT/'delivery_validation.json')
    assert delivery['status']=='PASS_DELIVERY' and delivery['v1_v2_results_equal']
    assert delivery['supplement_rows']==1104 and delivery['main_numeric_rows']==21
    assert load(OUT/'results.json')==load(OUT.parent/'pr0012_64bf0f2c7697_exploration_v1/results.json')
    primary=load(SRC/'second_path_validation.json')['primary']
    assert primary=={'SUPPORTED':12,'UNRESOLVED':1,'OPPOSITE':0,'TECHNICAL_UNESTABLISHED':0}
    link_count=0
    for document in [OUT/'PR0012_EXPLORATORY_INSIGHTS_JA.md',OUT/'PROTOCOL.md',ROOT/'audits/verification/2026-10-18_PR0012_exploration.md']:
        for link in re.findall(r'\]\(([^)]+)\)',document.read_text()):
            if link.startswith(('http:','https:','#')):continue
            path=Path(link) if link.startswith('/') else document.parent/link
            path=path.resolve()
            assert path.exists() or path==target, (str(document),link)
            link_count+=1
    files=[]
    for directory in [OUT,OUT.parent/'pr0012_64bf0f2c7697_exploration_v1']:
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT/'numerics/scripts'/name for name in ['explore_pr0012_saved.py','plot_pr0012_exploration_addendum.py','verify_pr0012_exploration.py','close_pr0012_exploration.py']]
    files += [ROOT/'audits/verification/2026-10-18_PR0012_exploration.md']
    outputs={str(p.relative_to(ROOT)):sha(p) for p in files}
    result=dict(status='PASS_CLOSEOUT',protected_sources=protected,outputs=outputs,current_navigation=navigation,
      **prefixes,primary=primary,tests=dict(saved_inputs=51,scalar_result_records=528,supplement_rows=1104,main_numeric_rows=21,figures_displayed=7,local_links_checked=link_count,old_anchor_count=len(snap['anchors']),navigation_removals_restore_old_hashes=True),
      archive_member_scan_rerun=False,raw_decisions_recomputed_this_turn=False,scientific_records_generated_locally=0,policy_changes=0,manuscript_edits=0,independent_reviewer=False,
      gaps=['post-hoc views of the same 50 repeats, not new confirmation','R05 remains unresolved','A/U shadows not deployed as a new gate','empirical E FPR not V certification','no sensitivity guarantee or resource optimization','no new physical or Colab experiment'])
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_CLOSEOUT',len(protected),'immutable anchors;',len(outputs),'new output/source anchors;',link_count,'links')

if __name__=='__main__':main()
