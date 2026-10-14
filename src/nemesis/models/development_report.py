"""Descriptive B03 metrics; no tuning, significance testing or later endpoints."""
import json
from pathlib import Path
import numpy as np
from .train import equal_weights,load_rows
from .calibrate import probabilities,ensemble_distribution
from nemesis.data.common import write_json
from .budget import B03Meter

def describe(probs,targets,rows):
    p=probs[np.arange(len(targets)),targets]; order=np.argsort(-probs,axis=1,kind='stable')[:,:3]
    confidence=probs.max(1); correct=(order[:,0]==targets).astype(float)
    scores=dict(nll=-np.log(np.maximum(p,1e-30)),top1=correct,top3=(order==targets[:,None]).any(1).astype(float),
                brier=(probs.astype(np.float64)**2).sum(1)-2*p+1)
    weights=equal_weights(rows)
    bins=np.minimum(9,(confidence*10).astype(int)); reliability=[]
    for b in range(10):
        ids=np.where(bins==b)[0]; mass=float(weights[ids].sum())
        reliability.append(dict(bin=b,weight=mass,decisions=len(ids),
            confidence=float(np.dot(confidence[ids],weights[ids])/mass) if mass else None,
            accuracy=float(np.dot(correct[ids],weights[ids])/mass) if mass else None))
    strata=[]
    for field in ('split','phase','mover_color'):
        for value in sorted({r[field] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r[field]==value]; w=equal_weights([rows[i] for i in ids])
            strata.append(dict(field=field,value=value,decisions=len(ids),
                **{k:float(np.dot(v[ids],w)) for k,v in scores.items()}))
    return dict(overall={k:float(np.dot(v,weights)) for k,v in scores.items()},strata=strata,reliability=reliability)

def build_development_report(out):
    out=Path(out); meter=B03Meter(out,gpu=False)
    try:
        rows,_=load_rows(Path('artifacts/B01'),'dev',('H1','H2'),out/'access_log.jsonl')
        registry=json.loads((out/'model_registry.json').read_text()); results={}; ensembles={}
        for r in registry['models']:
            if r['cohort']!='dev': continue
            family=r['family']; identity=r['identity']; mid=r['model_id']
            p=probabilities(np.load(out/'logits'/f'{mid}_H.npy',mmap_mode='r'),r['temperature'])
            if identity:
                g=probabilities(np.load(out/'logits'/f"{r['group']}_H.npy",mmap_mode='r'),r['temperature'])
                p=r['alpha']*p+(1-r['alpha'])*g
                if r['role']=='T': ensembles.setdefault(identity,[]).append((p,r['alpha']))
                ids=[i for i,row in enumerate(rows) if row['player_id']==identity]
            else: ids=list(range(len(rows)))
            targets=np.load(out/'logits'/f'{family}_H_targets.npy')
            results[mid]=describe(p[ids],targets[ids],[rows[i] for i in ids]); meter.checkpoint()
        for player,values in ensembles.items():
            p=ensemble_distribution([v[0] for v in values],[v[1] for v in values]); ids=[i for i,r in enumerate(rows) if r['player_id']==player]
            targets=np.load(out/'logits'/'maia3_H_targets.npy')
            results['T_ensemble_'+player]=describe(p[ids],targets[ids],[rows[i] for i in ids]); meter.checkpoint()
        write_json(out/'validation'/'descriptive_metrics.json',dict(models=results,cohort='dev',confirmation_H_opened=False,
            aggregation='mean decisions within game; mean games within person; mean people',significance='NOT_RUN'))
        # Aggregate descriptive plot for G0/G and the eight correct T ensembles.
        from html import escape
        colors=['#777','#2166ac','#b2182b']; lines=[]
        series=[('G0',results['G0']['reliability']),('G',results['G']['reliability'])]
        # Display per-player bins individually; averaging conditional bins across
        # people would distort the unconditional mass weighting.
        series.extend((k,v['reliability']) for k,v in results.items() if k.startswith('T_ensemble_'))
        for i,(name,values) in enumerate(series):
            points=' '.join(f"{60+400*v['confidence']:.2f},{450-400*v['accuracy']:.2f}" for v in values if v['weight']>0)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[min(2,i)]}" opacity="0.7"/><text x="490" y="{65+i*25}" font-size="13">{escape(name)}</text>')
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="720" height="510" viewBox="0 0 720 510"><rect width="720" height="510" fill="white"/><text x="60" y="25" font-size="18">Development move-confidence reliability</text><path d="M60 50 V450 H460 M60 450 L460 50" stroke="#bbb" fill="none"/><text x="210" y="485">Confidence (0 to 1)</text><text x="5" y="35">Accuracy</text>'+''.join(lines)+'</svg>'
        (out/'validation'/'calibration_curve.svg').write_text(svg,encoding='utf8')
        return results
    finally: meter.checkpoint(enforce=False)

if __name__=='__main__': build_development_report(Path('artifacts/B03'))
