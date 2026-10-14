"""Post-seal geometric diagnostics and development dispositions."""
import numpy as np

def headroom(q,generic_index):
    q=np.asarray(q,dtype=float) # safe move x target
    if q.ndim!=2 or not np.isfinite(q).all():raise ValueError('Incomplete headroom values')
    h=q.max(axis=0)-q[generic_index]
    return dict(H=h.tolist(),H_ID=float(q.max(axis=0).mean()-q.mean(axis=1).max()))

def pilot_decision(h_upper,n,qualified):
    if not qualified:return dict(challenge='INSUFFICIENT_EVIDENCE')
    skip=n>=128 and len(h_upper)==2 and all(x<.005 for x in h_upper)
    return dict(challenge='SKIPPED_NO_HEADROOM' if skip else 'BOUNDED_CHALLENGE_NEEDED')
