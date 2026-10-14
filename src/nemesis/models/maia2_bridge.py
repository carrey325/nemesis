"""Maia-2 Blitz current-board policy. Value head is never exposed."""
from pathlib import Path
import sys
import torch
from .common import model_record, rebuild, tensor_hash, distribution, lock_fp32
_SOURCE=Path('vendor/maia2-67bee6ce7e2f264344fe6d2026a2e18c76a82efa').resolve()
sys.path.insert(0,str(_SOURCE))
from maia2.main import MAIA2Model
from maia2.utils import parse_args,get_all_possible_moves,create_elo_dict,mirror_move
from maia2.train import load_model_state_dict
from maia2.inference import preprocessing,_masked_softmax


class Maia2Predictor:
    def __init__(self,checkpoint_path=None,lock_path=Path('artifacts/B00/source_lock.json'),device='cpu'):
        lock_fp32()
        self.record=model_record('maia2_blitz',lock_path); self.device=device
        if checkpoint_path is not None and Path(checkpoint_path).resolve()!=Path(self.record['path']).resolve():
            raise ValueError('Checkpoint must match locked path')
        if Path(sys.modules['maia2'].__file__).resolve().parent.parent!=_SOURCE:
            raise ValueError('Unexpected imported Maia2 source')
        self.cfg=parse_args(self.record['config_file']['path'])
        self.vocab=get_all_possible_moves(); self.mapping={m:i for i,m in enumerate(self.vocab)}
        self.elos=create_elo_dict(); torch.set_num_threads(1)
        self.model=MAIA2Model(len(self.vocab),self.elos,self.cfg)
        checkpoint=torch.load(self.record['path'],map_location='cpu',weights_only=True)
        load_model_state_dict(self.model,checkpoint['model_state_dict'])
        self.model.to(device).eval()

    def encode(self,context):
        board=rebuild(context)
        token,se,oe,mask=preprocessing(board.fen(),context.self_elo,context.oppo_elo,self.elos,self.mapping)
        return token,mask.bool(),board,se,oe

    @torch.inference_mode()
    def predict_many(self,contexts):
        encoded=[self.encode(c) for c in contexts]
        logits,_,_=self.model(torch.stack([e[0] for e in encoded]).to(self.device),
            torch.tensor([e[3] for e in encoded],device=self.device),
            torch.tensor([e[4] for e in encoded],device=self.device))
        masks=torch.stack([e[1] for e in encoded]).to(self.device)
        probs=_masked_softmax(logits.float(),masks).cpu()
        result=[]
        for i,(token,mask,board,se,oe) in enumerate(encoded):
            idx=mask.nonzero().flatten().tolist()
            moves=[self.vocab[j] if board.turn else mirror_move(self.vocab[j]) for j in idx]
            result.append(distribution(board,moves,probs[i,idx].tolist(),'Maia2-Blitz',self.record['sha256'],
                                       tensor_hash(token,mask,se,oe)))
        return result

    def predict(self,context): return self.predict_many([context])[0]
