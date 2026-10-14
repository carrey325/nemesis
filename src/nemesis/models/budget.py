"""Aggregate B03 safety stops across training, validation, tests and retries."""
import json,time
from pathlib import Path
from nemesis.data.common import write_json
from nemesis.preflight import ResourceLimit

class B03Meter:
    def __init__(self,out,gpu=True):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.cpu=time.process_time(); self.wall=time.perf_counter(); self.gpu=gpu
        path=self.out/'resource_ledger.json'
        old=json.loads(path.read_text()) if path.exists() else {}
        self.prior_cpu=old.get('measured_run_cpu_seconds',0); self.prior_gpu=old.get('measured_run_gpu_seconds',0)
        self.last_disk=0.; self.disk=old.get('disk_bytes',14221799641)

    def checkpoint(self,enforce=True):
        path=self.out/'test_resource_receipts.jsonl'
        tests=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        cpu=self.prior_cpu+time.process_time()-self.cpu
        gpu=self.prior_gpu+(time.perf_counter()-self.wall if self.gpu else 0)
        if time.perf_counter()-self.last_disk>60:
            self.disk=sum(p.stat().st_size for p in Path('.').rglob('*') if p.is_file()); self.last_disk=time.perf_counter()
        total_cpu=cpu+sum(t['cpu_seconds'] for t in tests)+360
        total_gpu=gpu+sum(t['gpu_seconds'] for t in tests)
        value=dict(stage='B03',measured_run_cpu_seconds=cpu,measured_run_gpu_seconds=gpu,
            test_cpu_seconds=sum(t['cpu_seconds'] for t in tests),test_gpu_seconds=sum(t['gpu_seconds'] for t in tests),
            outside_process_cpu_allowance_hours=.1,cpu_core_hours=total_cpu/3600,gpu_hours=total_gpu/3600,
            project_cpu_core_hours=4.424717881944445+total_cpu/3600,project_gpu_hours=.19443688016675878+total_gpu/3600,
            stage_cpu_cap=4,stage_gpu_cap=8,project_cpu_cap=48,project_gpu_cap=20,disk_bytes=self.disk,
            disk_gib=self.disk/2**30,disk_cap_gib=80,external_spend_usd=0,
            measurement='process_time (all threads); no child workers; full runner resident-CUDA wall, tests conservatively full wall; 360s setup/CLI/report allowance')
        write_json(self.out/'resource_ledger.json',value)
        if enforce and (total_cpu>=3.9*3600 or total_gpu>=7.9*3600 or self.disk>=79*2**30):
            raise ResourceLimit('B03 safety stop before 4 CPU-core-hour / 8 GPU-hour / 80 GiB caps')
        return value
