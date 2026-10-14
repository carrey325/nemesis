"""Game-based, same-identity source/target paired descriptive statistics."""
from collections import Counter
import numpy as np

def frequency(statuses):
    n=len(statuses);e=statuses.count('ELIGIBLE');m=sum(s in ('MISSING','BLOCKED','UNSCREENED') for s in statuses)
    if not n:raise ValueError('Empty natural denominator')
    return dict(denominator=n,eligible=e,missing=m,eligible_rate=e/n,lower=e/n,upper=(e+m)/n)

def independent_games(rows):return len({r['source_game_id'] for r in rows})

def matrix_summary(matrix):
    m=np.asarray(matrix,float);n=len(m)
    if m.shape!=(n,n) or n<2 or not np.isfinite(m).all():raise ValueError('Incomplete square matrix')
    d=np.diag(m);wrong=(m.sum(axis=0)-d)/(n-1)
    return dict(G=float(d.mean()),S=float((d-wrong).mean()),per_target_G=d.tolist(),
        per_target_S=(d-wrong).tolist(),row_transfer=m.mean(axis=1).tolist(),
        off_diagonal_mean=float((m.sum()-d.sum())/(n*(n-1))),
        off_diagonal_above_005=float((m[~np.eye(n,dtype=bool)]>=.005).mean()))

def two_panel_yield(f,g,w):return float(np.sum(np.asarray(f)*np.asarray(g)*np.asarray(w)))

def summarize_matrix(records,generic_records,identities,weights):
    # Each record includes selection identity; AuditRecord alone omits that axis.
    baseline={(r['root_id'],r['target_id']):r['q'] for r in generic_records}
    m=np.zeros((len(identities),len(identities)));norm=np.zeros_like(m)
    for r in records:
        i=identities.index(r['identity']);j=identities.index(r['target_id']);w=weights[r['root_id']]
        m[i,j]+=w*(r['q']-baseline[(r['root_id'],r['target_id'])]);norm[i,j]+=w
    if (norm==0).any():raise ValueError('Missing matrix cells')
    m/=norm
    return dict(matrix=m.tolist(),**matrix_summary(m))

def bootstrap_endpoints(records,panels,config):
    """records contains per-root target gain/spec/H and useful/specific flags.

    Same identity draw weights both target and source; wrong rows have already
    been contrasted against the original other seven policies, never relabeled.
    Independent N/S game draws are paired across metrics and evaluator families.
    """
    identities=config['identities'];rng=np.random.default_rng(8675309)
    reps=config.get('replicates',5000);groups={};ngroups={}
    for row in records:groups.setdefault(row['stratum'],[]).append(row)
    for row in panels:ngroups.setdefault(row['sampling_stratum'],[]).append(row['status']=='ELIGIBLE')
    arrays={h:np.array([r['vectors'] for r in rows]) for h,rows in groups.items()}
    oracle={}
    for h,rows in groups.items():
        for family in ('E3','E2'):
            key='safe_q_'+family
            if all(key in r for r in rows):
                a=np.zeros((len(rows),max(len(r[key]) for r in rows),8))
                for i,r in enumerate(rows):a[i,:len(r[key])]=r[key]
                oracle[h,family]=a
    for rows in groups.values():
        if independent_games(rows)!=len(rows):raise ValueError('Repeated game in bootstrap source stratum')
    ngroups={h:np.array(v) for h,v in ngroups.items()}
    metrics=['E3_G','E3_S','E2_G','E2_S','E3_H','E2_H','Y_E3','Y_E2','F_useful','F_specific','E3_H_ID','E2_H_ID']
    results={}
    for mode in ('same_ID_block','fixed_identity_game_only'):
        samples=np.zeros((reps,len(metrics)))
        for b in range(reps):
            mult=np.ones(8) if mode=='fixed_identity_game_only' else np.bincount(rng.integers(0,8,8),minlength=8)
            tw=mult/mult.sum();total=np.zeros(len(metrics));sw=0.
            for h,rows in groups.items():
                source=identities.index(h.split('|')[0]);weight=mult[source]
                if not weight:continue
                arr=arrays[h] # roots x 8 metrics x 8 targets
                draw=rng.integers(0,len(arr),len(arr))
                means=arr[draw].mean(axis=0)@tw
                n=ngroups[h];f=n[rng.integers(0,len(n),len(n))].mean()
                hid=means[8:10].copy() if len(means)>=10 else np.zeros(2)
                for i,family in enumerate(('E3','E2')):
                    if (h,family) in oracle:
                        oq=oracle[h,family][draw]
                        hid[i]=float(((oq.max(axis=1)@tw)-(oq@tw).max(axis=1)).mean())
                total+=weight*np.r_[means[:6],f*means[0],f*means[2],f*means[6:8],hid];sw+=weight
            samples[b]=total/sw
        results[mode]={k:dict(ci95=np.quantile(samples[:,i],[.025,.975]).tolist(),
          upper_975=float(np.quantile(samples[:,i],.975))) for i,k in enumerate(metrics)}
    return dict(results=results,replicates=reps,seed=8675309,identities=identities,
      algorithm='One identity multiplicity weights target and source; source/window/color games sampled within each panel; N/S independently resampled; fixed wrong-policy library',
      independent_S_games=independent_games(records),players=8)
