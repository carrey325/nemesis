"""Calibration-only fixed grids and preregistered B03 development gate."""
import numpy as np

def require_calibration(rows):
    if not rows or any(r['split']!='calibration' for r in rows): raise ValueError('Calibration-only selection')

def probabilities(logits,temperature):
    x=np.asarray(logits,dtype=np.float64)/temperature
    x=x-np.max(x,axis=-1,keepdims=True); x=np.exp(x)
    return x/x.sum(-1,keepdims=True)

def weighted_nll(probs,targets,weights):
    return float(np.sum(-np.log(np.maximum(probs[np.arange(len(targets)),targets],1e-30))*weights))

def choose_temperature(logits,targets,weights,grid):
    scores=[{'temperature':float(t),'nll':weighted_nll(probabilities(logits,t),targets,weights)} for t in grid]
    return {**min(scores,key=lambda r:(r['nll'],r['temperature'])),'candidates':scores}

def choose_alpha(personal,group,targets,weights,grid):
    scores=[{'alpha':float(a),'nll':weighted_nll(a*personal+(1-a)*group,targets,weights)} for a in grid]
    return {**min(scores,key=lambda r:(r['nll'],r['alpha'])),'candidates':scores,'included':True}

def ensemble_distribution(values,alphas):
    if len(values)!=3 or len(alphas)!=3: raise ValueError('T ensemble requires all three fixed seeds')
    if all(a==0 for a in alphas):
        if not all(np.array_equal(values[0],v) for v in values[1:]): raise ValueError('Alpha-zero seeds must share identical group policy')
        return values[0]
    return np.mean(values,axis=0,dtype=np.float64)

def development_gate(t_improvements,e3_improvement,e2_improvement):
    if len(t_improvements)!=8: raise ValueError('Gate requires all eight development players')
    mean=float(np.mean(t_improvements)); improved=sum(v>0 for v in t_improvements)
    passed=mean>0 and improved>=4 and e3_improvement>0 and e2_improvement>0
    return dict(passed=bool(passed),research_status='DEV_SIGNAL_FOUND' if passed else 'DEV_SIGNAL_NOT_FOUND',
        t_mean_nll_improvement=mean,t_players_improved=improved,e3_nll_improvement=float(e3_improvement),
        e2_nll_improvement=float(e2_improvement),confirmation_status='TRAIN_AFTER_GATE' if passed else 'NOT_TRAINED')

def qualified_development_gate(t,e3_group,e3_base,e2_group,e2_base):
    result=development_gate(t,e3_group,e2_group)
    result.update(b03_group_gate_passed=result['passed'],E3_vs_group=float(e3_group),E3_vs_base=float(e3_base),
                  E2_vs_group=float(e2_group),E2_vs_base=float(e2_base),E2_base_qualified=bool(e2_base>0))
    result['passed']=bool(result['passed'] and result['E2_base_qualified'])
    result['research_status']='DEV_SIGNAL_FOUND' if result['passed'] else 'DEV_SIGNAL_NOT_FOUND'
    result['confirmation_status']='TRAIN_AFTER_GATE' if result['passed'] else 'NOT_TRAINED'
    result['rule']='B03 T/group and E/group rules; additional E2 base qualification from M3; E3 base comparison diagnostic'
    return result

def calibrate(checkpoint,calibration_path,config):
    """I3 public wrapper; training selection functions above remain unchanged."""
    from .calibration_api import calibrate as implementation
    return implementation(checkpoint,calibration_path,config)
