"""Shared finite candidates and exhaustive reply leaves, always root POV."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
import chess
import chess.engine
import psutil
from nemesis.data.common import digest
from .types import CandidateBundle

def hash_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def rebuild_root(root):
    b=chess.Board(root.initial_fen)
    if not b.is_valid(): raise ValueError('Invalid initial FEN')
    for m in root.history_uci:
        if chess.Move.from_uci(m) not in b.legal_moves: raise ValueError('Illegal prefix')
        b.push_uci(m)
    if b.fen()!=root.fen6 or b.turn!=root.root_color: raise ValueError('Root prefix/FEN/POV mismatch')
    return b

def terminal_value(board,color):
    outcome=board.outcome(claim_draw=True)
    if outcome is None: return None
    return .5 if outcome.winner is None else float(outcome.winner==color)

def score_value(score,color):
    s=score.pov(color)
    wdl=s.wdl(model='sf',ply=30)
    return s.score(),(wdl.wins+.5*wdl.draws)/1000

def safe_moves(cp):
    finite={m:v for m,v in cp.items() if v is not None and math.isfinite(v)}
    if not finite: return ()
    best=max(finite.values())
    return tuple(sorted(m for m,v in finite.items() if best-v<=25))

def expected_value(probabilities,values):
    if set(probabilities)!=set(values): raise ValueError('Incomplete legal reply mass')
    if any(not math.isfinite(p) or p<0 for p in probabilities.values()) or abs(sum(probabilities.values())-1)>1e-6:
        raise ValueError('Invalid complete probability distribution')
    if any(not math.isfinite(u) or not 0<=u<=1 for u in values.values()): raise ValueError('Invalid U')
    # Full legal mass only. Correct permitted floating-point softmax roundoff;
    # never drop a tail or renormalize a top-k subset.
    return math.fsum(probabilities[r]*values[r] for r in values)/math.fsum(probabilities.values())

class CandidateCache:
    """Verified engine bundles; probability caches never enter this namespace."""
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
    @staticmethod
    def key(root,engine_hash,budget_hash):
        from dataclasses import asdict
        return hash_object(['B04A_ENGINE_CACHE_V1',asdict(root),engine_hash,budget_hash])
    def put(self,root,bundle):
        from dataclasses import asdict
        from nemesis.data.common import write_json
        if root.root_id!=bundle.root_id: raise ValueError('Cache root mismatch')
        key=self.key(root,bundle.engine_hash,bundle.budget_hash);body=asdict(bundle)
        write_json(self.path/(key+'.json'),dict(key=key,body=body,body_hash=hash_object(body)))
    def get(self,root,engine_hash,budget_hash):
        key=self.key(root,engine_hash,budget_hash);path=self.path/(key+'.json')
        if not path.exists():return None
        value=json.loads(path.read_text())
        if value['key']!=key or hash_object(value['body'])!=value['body_hash']:raise ValueError('Corrupt engine cache')
        body=value['body']
        if body['root_id']!=root.root_id or body['engine_hash']!=engine_hash or body['budget_hash']!=budget_hash:
            raise ValueError('Cache lineage mismatch')
        body['moves']=tuple(body['moves']);body['safe_moves']=tuple(body['safe_moves'])
        return CandidateBundle(**body)

class EngineSession:
    def __init__(self,config):
        e=config['engine']
        fixed=dict(threads=1,hash_mb=128,candidate_k=5,nomination_nodes=100000,
          forced_candidate_nodes=50000,reply_nodes=20000,audit_nodes=200000,sensitivity_nodes=400000,safety_delta_cp=25)
        if any(e.get(k)!=v for k,v in fixed.items()): raise ValueError('Engine contract changed')
        self.config=config
        record=json.loads(Path('artifacts/B00/source_lock.json').read_text())['stockfish']
        for r in [record,*record['nnue_files']]:
            if digest(r['path'])!=r['sha256']: raise ValueError('B00 engine/NNUE hash mismatch')
        self.engine_hash=hash_object(record)
        self.engine=chess.engine.SimpleEngine.popen_uci(str(Path(record['path']).resolve()),timeout=120)
        self.process=psutil.Process(self.engine.transport.get_pid())
        self.engine.configure({'Threads':1,'Hash':128,'Skill Level':20,'UCI_LimitStrength':False,
          'UCI_ShowWDL':True,'SyzygyPath':''})
        self.cpu_start=self.cpu_seconds(); self.nodes=0; self.calls=[]; self.cache={}
    def cpu_seconds(self):
        t=self.process.cpu_times(); return t.user+t.system
    def close(self):
        self.final_cpu=self.cpu_seconds()-self.cpu_start
        self.engine.quit()
    def analyse(self,b,color,nodes,root_moves=None,multipv=1,namespace='decision',cached=False):
        key=hash_object([namespace,b.root().fen(),[m.uci() for m in b.move_stack],color,nodes,
          [m.uci() for m in root_moves] if root_moves else None,multipv,self.engine_hash])
        if cached and key in self.cache: return self.cache[key]
        self.engine.configure({'Clear Hash':None})
        tick=time.perf_counter(); cpu=self.cpu_seconds()
        infos=self.engine.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=root_moves,multipv=multipv)
        if isinstance(infos,dict): infos=[infos]
        if not infos or any(not x.get('pv') or 'score' not in x or 'wdl' not in x for x in infos):
            raise RuntimeError('Incomplete Stockfish output')
        actual=max(x.get('nodes',0) for x in infos); self.nodes+=actual
        self.calls.append(dict(namespace=namespace,requested_nodes=nodes,actual_nodes=actual,
          cpu_seconds=self.cpu_seconds()-cpu,wall_seconds=time.perf_counter()-tick,cache_key=key))
        values=[]
        for info in infos:
            s=info['score'].pov(color); w=info['wdl'].pov(color)
            values.append(dict(move=info['pv'][0].uci(),cp=s.score(),mate=s.mate(),u=(w.wins+.5*w.draws)/1000))
        if cached: self.cache[key]=values
        return values

def build_candidate_bundle(root,config):
    session=config.get('_engine_session'); own=session is None
    if own: session=EngineSession(config)
    try:
        b=rebuild_root(root); budget_hash=hash_object(config['engine'])
        if b.is_game_over(claim_draw=True):
            return CandidateBundle(root.root_id,(),(),{}, {},session.engine_hash,budget_hash,'TERMINAL')
        proposed=session.analyse(b,root.root_color,100000,multipv=min(5,b.legal_moves.count()))
        moves=tuple(x['move'] for x in proposed)
        cp={m:session.analyse(b,root.root_color,50000,[chess.Move.from_uci(m)])[0]['cp'] for m in moves}
        safe=safe_moves(cp)
        if any(v is None for v in cp.values()): status='MATE'
        elif min(abs(v) for v in cp.values())>100 or abs(max(cp.values()))>100: status='OUT_OF_RANGE'
        elif len(safe)<2: status='NO_BRANCH'
        else: status='ELIGIBLE'
        values={}
        if not config.get('_screen_only',False):
            values=reply_leaves(root,safe or moves[:1],session,20000,'decision')
        return CandidateBundle(root.root_id,moves,safe,cp,values,session.engine_hash,budget_hash,status)
    finally:
        if own: session.close()

def reply_leaves(root,moves,session,nodes,namespace):
    b=rebuild_root(root); result={}
    for m in moves:
        branch=b.copy(); branch.push_uci(m); leaves={}
        if terminal_value(branch,root.root_color) is not None:
            leaves['__terminal__']=terminal_value(branch,root.root_color)
        else:
            for r in sorted(x.uci() for x in branch.legal_moves):
                leaf=branch.copy(); leaf.push_uci(r)
                u=terminal_value(leaf,root.root_color)
                if u is None: u=session.analyse(leaf,root.root_color,nodes,namespace=namespace,cached=True)[0]['u']
                leaves[r]=u
        result[m]=leaves
    return result

def audit_bundle(root,bundle,session,nodes=200000,*,selection_seal=None,selections=None,traces=None,benchmark=False):
    if not benchmark:
        from .evaluate import verify_selection_seal
        if selection_seal is None: raise PermissionError('Scoring audit requires sealed selections first')
        verify_selection_seal(selections,selection_seal,traces=traces,require_trace=True)
    b=rebuild_root(root)
    selected=tuple(s.move_uci for s in (selections or []) if s.root_id==root.root_id)
    scored=tuple(dict.fromkeys([*(bundle.safe_moves or bundle.moves[:1]),*selected]))
    candidates=tuple(dict.fromkeys([*bundle.moves,*selected]))
    cp={m:session.analyse(b,root.root_color,nodes,[chess.Move.from_uci(m)],namespace='audit')[0]['cp'] for m in candidates}
    return cp,reply_leaves(root,scored,session,nodes,'audit')
