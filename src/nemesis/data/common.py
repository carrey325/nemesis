import hashlib
import json
import os
import time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def append_json(path,value):
    with Path(path).open('a',encoding='utf-8') as f: f.write(json.dumps(value,sort_keys=True)+'\n')

class ResourceLimit(RuntimeError): pass

class Meter:
    """Direct process CPU accounting; B01 does not spawn children or use GPU."""
    def __init__(self,out):
        self.out=Path(out); self.start=time.process_time(); self.prior=0
        p=self.out/'resource_ledger.json'
        if p.exists(): self.prior=json.loads(p.read_text())['cpu_seconds']
    def checkpoint(self):
        cpu=self.prior+time.process_time()-self.start
        size=sum(p.stat().st_size for p in Path('.').rglob('*') if p.is_file() and '.git' not in p.parts)
        write_json(self.out/'resource_ledger.json',{'cpu_seconds':cpu,'cpu_core_hours':cpu/3600,'charged_cpu_core_hours':cpu/3600+0.05,'gpu_hours':0,'disk_bytes':size,'external_spend_usd':0,'prior_B00_cpu_core_hours':0.25,'prior_B00_gpu_hours':0.01,'project_cpu_core_hours':0.25+cpu/3600+0.05,'measurement':'time.process_time for pipeline; no pipeline child processes; GPU unused; outside-pipeline receipts and conservative allowance retained separately','setup_and_tool_allowance_cpu_core_hours':0.05})
        if cpu/3600+0.05>=3.9 or size>=79*1024**3: raise ResourceLimit('B01 resource safety stop before hard cap')
