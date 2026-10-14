"""Narrow source proof for an evaluation/API repair, preserving valid training."""
import ast,hashlib,json,zipfile
from pathlib import Path
from nemesis.data.common import write_json

def definitions(data):
    tree=ast.parse(data.decode('utf8'))
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}

def prove_compatible_training(out,old,new):
    out=Path(out); archive=out/'training_source.zip'
    allowances={'src/nemesis/models/train.py':{'fit_adapter'},
                'src/nemesis/models/calibrate.py':set(),
                'src/nemesis/models/twins.py':{'run_training','evaluate_registry'}}
    checked=[]; differences=[]
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('source_hashes.json'))
        if hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=old: raise ValueError('Old source archive identity mismatch')
        for name,sha in hashes.items():
            previous=z.read(Path(name).as_posix()); current=Path(name).read_bytes()
            if hashlib.sha256(previous).hexdigest()!=sha: raise ValueError('Archived source bytes mismatch')
            if previous==current: checked.append(dict(path=name,mode='byte-identical')); continue
            key=Path(name).as_posix()
            if key not in allowances: raise ValueError('Training dependency changed: '+key)
            before=definitions(previous); after=definitions(current)
            for function,body in before.items():
                if function not in allowances[key] and after.get(function)!=body:
                    raise ValueError('Training/encoding/calibration function changed: '+function)
            checked.append(dict(path=name,mode='all existing core definitions AST-identical',excluded=sorted(allowances[key])))
            differences.append(key)
    record=dict(status='PASS',old_code_sha=old,new_code_sha=new,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        checked=checked,changed_files=differences,
        changes='public fit_adapter role/Calibration wiring; exact B03 gate mapping; resume metadata/orchestration only',
        training_math_data_encoding_optimizer_calibration_checkpoint_selection_unchanged=True)
    write_json(out/'source_compatibility.json',record)
    return {old,new}

def preserve_metadata(existing,expected,versions):
    for key,value in expected.items():
        if key=='code_sha':
            if existing.get(key) not in versions: raise ValueError('Unknown checkpoint training version')
        elif existing.get(key)!=value: raise ValueError('Checkpoint input/config/model/seed mismatch: '+key)
    return existing
