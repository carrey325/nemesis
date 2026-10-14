"""B02 process CPU and conservative resident-GPU accounting; no scheduler."""
import json
from pathlib import Path
import time
from nemesis.data.common import write_json
from nemesis.preflight import ResourceLimit


class B02Meter:
    def __init__(self,out):
        self.out=Path(out); self.cpu_start=time.process_time(); self.gpu_start=None
        self.gpu_seconds=0
        self.previous_cpu=0; self.previous_gpu=0
        path=self.out/'resource_ledger.json'
        if path.exists():
            old=json.loads(path.read_text()); self.previous_cpu=old['measured_run_cpu_seconds']
            self.previous_gpu=old['measured_run_gpu_seconds']

    def gpu_on(self): self.gpu_start=time.perf_counter()

    def gpu_off(self):
        if self.gpu_start is not None:
            self.gpu_seconds+=time.perf_counter()-self.gpu_start; self.gpu_start=None

    def checkpoint(self, enforce=True):
        test_path=self.out/'test_resource_receipts.jsonl'
        test_receipts=[json.loads(line) for line in test_path.read_text().splitlines()] if test_path.exists() else []
        tests=sum(r['cpu_seconds'] for r in test_receipts)
        test_gpu=sum(r.get('gpu_seconds',0) for r in test_receipts)
        cpu=self.previous_cpu+time.process_time()-self.cpu_start
        gpu=self.previous_gpu+self.gpu_seconds+(time.perf_counter()-self.gpu_start if self.gpu_start is not None else 0)
        total_cpu=(cpu+tests)/3600+.05
        value=dict(measured_run_cpu_seconds=cpu,measured_run_gpu_seconds=gpu,test_cpu_seconds=tests,
                   test_gpu_seconds=test_gpu,cpu_core_hours=total_cpu,gpu_hours=(gpu+test_gpu)/3600,external_spend_usd=0,
                   outside_run_cpu_allowance_hours=.05,stage_cpu_cap=1,stage_gpu_cap=.5,
                   prior_B00_cpu_core_hours=.25,prior_B00_gpu_hours=.01,
                   prior_B01_cpu_core_hours=3.9177734375,prior_B01_gpu_hours=0,
                   project_cpu_core_hours=.25+3.9177734375+total_cpu,
                   project_gpu_hours=.01+(gpu+test_gpu)/3600,
                   measurement='process_time including all threads; tests separately; GPU wall time while model resident, including encoding/waits; no child workers')
        write_json(self.out/'resource_ledger.json',value)
        if enforce and (total_cpu>=.95 or gpu+test_gpu>=1740):
            raise ResourceLimit('B02 safety stop before 1 CPU-core-hour / 0.5 GPU-hour hard cap')
        return value
