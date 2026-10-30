# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Prespecified path-conditioned accounting, not a new policy or optimization."""
import json
import numpy as np
from report_pr0011_recovery import OUT,S,ratio,dump

def main():
    d={}
    for k in ['actions','counts','paired','base']:
        d[k]=np.stack([np.load(OUT/'recomputed'/f'rep_{r:02d}.npz')[k] for r in range(50)])
    rows=[]
    for g,group in enumerate(S['groups']):
        for a,alpha in enumerate(S['alphas']):
            acts=d['actions'][:,g,36,2,a,2]
            for e,ev in enumerate(S['evaluations']):
                n=20000 if e==0 else 5000
                j=d['paired'][:,g,36,2,a,2,e]
                rec=acts==2;defer=acts==3
                raw=d['counts'][:,g,36,2,a,2,e,0]
                rows.append(dict(group=group,alpha=alpha,evaluation=ev,candidate='GMGPP112_n800',gate='G-MC',
                    recovery_replicates=int(rec.sum()),defer_replicates=int(defer.sum()),
                    recovery_issued=ratio(raw*rec,n*rec),
                    C0_switch_gain=int(j[rec,0,1:].sum()),C0_switch_loss=int(j[rec,1:,0].sum()),
                    defer_lost_P0_alarm=int(j[defer,2,0].sum()),defer_lost_P0_noalarm=int(j[defer,2,1].sum()),
                    planned_records=50*n,
                    unconditional_C0_rate=float(d['base'][:,g,0,e,a,2,1:].sum()/(50*n))))
    dump(OUT/'recovery_strata_triple800.json',rows)
    lines=['# 回復した反復に条件づけた率と未判定の分解','',
        '固定のG-MC、GM:GP:P=200:200:400、P2−P0。回復反復はVで選択され、Eは独立。提供時率は外側50反復の比で、提供反復が少ない区間は不安定。選択前後の母集団は同じではない。','',
        '|群|α|評価|C0回復反復|回復時警報%|95%区間%|回復時追加警報|回復時喪失警報|DEFERへ失ったP0警報|DEFERへ失ったP0非警報|全予定N|',
        '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for x in rows:
        p=x['recovery_issued'];rate='NA' if p['value'] is None else f"{p['value']*100:.4f}"
        ci='NA' if p['ci95'] is None else ', '.join(f'{v*100:.4f}' for v in p['ci95'])
        lines.append('|'+ '|'.join(map(str,[x['group'],x['alpha'],x['evaluation'],x['recovery_replicates'],rate,ci,x['C0_switch_gain'],x['C0_switch_loss'],x['defer_lost_P0_alarm'],x['defer_lost_P0_noalarm'],x['planned_records']]))+'|')
    (OUT/'RECOVERY_STRATA.md').write_text('\n'.join(lines)+'\n')
    print('STRATA PASS',len(rows))
if __name__=='__main__':main()
