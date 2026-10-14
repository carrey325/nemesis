"""Catch exclusion of alpha-zero players and future-label temperature tuning."""
import numpy as np
import pytest

def api():
    from nemesis.models import calibrate
    return calibrate

def test_alpha_zero_is_valid_not_player_drop():
    p=np.array([[.1,.9],[.1,.9]]); g=np.array([[.8,.2],[.8,.2]])
    result=api().choose_alpha(p,g,np.array([0,0]),np.array([.5,.5]),[0,.25,.5,.75,1])
    assert result['alpha']==0
    assert result['included'] is True

def test_calibration_rejects_future_split():
    with pytest.raises(ValueError): api().require_calibration([{'split':'H1'}])

def test_temperature_equal_game_player_nll():
    logits=np.array([[2.,0.],[0.,2.]])
    result=api().choose_temperature(logits,np.array([0,1]),np.array([.5,.5]),[.75,1,1.25])
    assert result['temperature']==.75

def test_dev_gate_negative_is_successful_methodological_outcome():
    result=api().development_gate([.1,.1,.1,-.01,-.01,-.01,-.01,-.01],.02,-.001)
    assert result['research_status']=='DEV_SIGNAL_NOT_FOUND'
    assert result['confirmation_status']=='NOT_TRAINED'

def test_zero_alpha_ensemble_is_exact_group_without_roundoff_signal():
    g=np.array([[.123456789012,.876543210988]],dtype=np.float64)
    result=api().ensemble_distribution([g.copy(),g.copy(),g.copy()],[0,0,0])
    assert np.array_equal(result,g)
    assert api().weighted_nll(result,np.array([1]),np.array([1.]))==api().weighted_nll(g,np.array([1]),np.array([1.]))

def test_e3_raw_base_is_diagnostic_not_extra_gate_condition():
    result=api().qualified_development_gate([.01]*8,.01,-.02,.01,.005)
    assert result['passed'] is True
    assert result['E3_vs_base']==-.02
    assert result['b03_group_gate_passed'] is True

def test_e2_still_needs_m3_raw_base_qualification():
    result=api().qualified_development_gate([.01]*8,.01,.02,.01,-.005)
    assert result['b03_group_gate_passed'] is True
    assert result['passed'] is False
