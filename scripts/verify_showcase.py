"""Check published arithmetic, copied-source hashes and local Markdown targets."""
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/"evidence/results.json").read_text(encoding="utf-8"))
    provenance=json.loads((ROOT/"evidence/provenance.json").read_text(encoding="utf-8"))
    people=data["prediction"]["players"]
    assert len(people)==8 and len({p["player"] for p in people})==8
    gains=[p["group_nll"]-p["personal_nll"] for p in people]
    assert all(g>0 for g in gains)
    assert math.isclose(sum(gains)/8, .04682818933853605,abs_tol=1e-12)
    matrix=data["prediction"]["identity_gain_matrix"]
    for j,p in enumerate(people):
        assert p["personal_rank"]==1
        assert matrix[j][j]==max(row[j] for row in matrix)
        assert math.isclose(matrix[j][j],p["nll_gain"],abs_tol=1e-12)
        wrong=sum(matrix[i][j] for i in range(8) if i!=j)/7
        assert math.isclose(matrix[j][j]-wrong,p["wrong_identity_mean_nll"]-p["personal_nll"],abs_tol=1e-12)
    assert math.isclose(sum(p["wrong_identity_mean_nll"]-p["personal_nll"] for p in people)/8,.0845312490210839,abs_tol=1e-12)
    for family,expected in (("E3",.0014716199503305466),("E2",.000007135176879187179)):
        effect=data["decision"][family]
        assert math.isclose(sum(effect["gain_by_player"])/8,effect["gain"],abs_tol=1e-12)
        assert math.isclose(effect["gain"],expected,abs_tol=1e-12)
        for key in ("ci95_fixed_players","ci95_player_and_game"):
            assert effect[key][0] <= effect["gain"] <= effect[key][1]
    assert data["decision"]["E3"]["ci95_fixed_players"][0]>0
    assert data["decision"]["E3"]["ci95_player_and_game"][0]<0
    assert data["decision"]["E2"]["ci95_fixed_players"][0]<0
    assert len(data["calibration_histories"])==24
    assert all(r["stopping"]=="patience" for r in data["calibration_histories"])
    q=data["lora_engineering"]
    assert q["trainable_parameters"]==16384 and q["samples"]==32
    assert q["frozen_base_unchanged"] and q["optimizer_rng_step_restored"]
    assert q["resume_parameter_error"]==0 and q["overfit"]["final_loss"]<q["overfit"]["initial_loss"]
    for record in provenance["unmodified_source_files"]:
        assert hashlib.sha256((ROOT/record["path"]).read_bytes()).hexdigest()==record["sha256"],record["path"]
    links=0
    for path in [ROOT/"README.md",*(ROOT/"docs").glob("*.md")]:
        text=path.read_text(encoding="utf-8")
        assert text.count("```")%2==0,path
        for target in re.findall(r"\]\(([^)]+)\)|src=\"([^\"]+)\"",text):
            dest=next(t for t in target if t)
            if dest.startswith(("https://","http://","#")):continue
            assert (path.parent/dest.split("#")[0]).exists(),(path,dest)
            links+=1
    for path in (ROOT/"assets").glob("*.svg"):ET.parse(path)
    assert len(list((ROOT/"assets").glob("*.svg")))==8
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(x.startswith(".") for x in path.relative_to(ROOT).parts):continue
        assert path.suffix not in (".pt",".pth",".pgn",".parquet",".npy",".npz",".safetensors"),path
        if path.suffix in (".json",".md",".py",".toml",".yml"):
            body=path.read_text(encoding="utf-8")
            assert not re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",body),path
            assert not re.search(r"[A-Z]:[\\/](?:Users|Python file)[\\/]",body),path
    print(f"PASS: 8-player arithmetic, both evaluator intervals, 24 histories, LoRA evidence, {len(provenance['unmodified_source_files'])} source hashes, {links} local links, 8 SVGs, publication allowlist.")


if __name__=="__main__":main()
