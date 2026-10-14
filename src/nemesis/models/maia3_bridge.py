"""Maia-3 23M FP32 policy; official vocabulary and per-position history view."""
from collections import deque
from pathlib import Path
import sys
import torch
from .common import model_record, rebuild, tensor_hash, distribution, lock_fp32

_SOURCE=Path('vendor/maia3-1e13597c42d4858b7cfd7cfdae01e297263364b2').resolve()
sys.path.insert(0,str(_SOURCE))
from maia3.utils import get_all_possible_moves, mirror_move
from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
from maia3.uci import parse_args
from maia3.models import MAIA3Model
_VOCAB = get_all_possible_moves()
_MAP = {m:i for i,m in enumerate(_VOCAB)}


def vocabulary(): return list(_VOCAB)


def config_for(device, checkpoint='weights/maia3-23m.pt'):
    return parse_args(['--model','maia3-23m','--checkpoint-path',str(checkpoint),
                       '--device',device,'--no-use-amp','--use-uci-history'])


def encode_context(context,cfg):
    board=rebuild(context)
    # Pop from the verified replay stack: only the last cfg.history positions
    # need tokenizing. Each is independently oriented, exactly as cmd_position.
    history=[]
    prior=board.copy()
    for _ in range(cfg.history):
        history.append(tokenize_board(prior))
        if not prior.move_stack: break
        prior.pop()
    history=deque(reversed(history),maxlen=cfg.history)
    tokens=get_historical_tokens(history,cfg,0,0,0,0)
    mask=get_legal_moves_mask(board,_MAP)
    return tokens,mask,board


class Maia3Predictor:
    def __init__(self, checkpoint_path=None, lock_path=Path('artifacts/B00/source_lock.json'), device='cpu'):
        lock_fp32()
        record=model_record('maia3_23m',lock_path)
        if checkpoint_path is not None and Path(checkpoint_path).resolve()!=Path(record['path']).resolve():
            raise ValueError('Checkpoint must match locked path')
        self.record=record; self.device=device
        self.cfg=config_for(device,record['path'])
        if any(getattr(self.cfg,k)!=v for k,v in record['config'].items()):
            raise ValueError('Upstream preset differs from B00 model lock')
        if Path(sys.modules['maia3'].__file__).resolve().parent.parent!=_SOURCE:
            raise ValueError('Unexpected imported Maia3 source')
        torch.set_num_threads(1)
        self.model=MAIA3Model(self.cfg)
        checkpoint=torch.load(record['path'],map_location='cpu',weights_only=True)
        state=checkpoint.get('model_state_dict',checkpoint)
        self.model.load_state_dict({k.replace('smolgen','gab'):v for k,v in state.items()},strict=True)
        self.model.to(device).eval()
        self.vocab=vocabulary()

    def encode(self,context): return encode_context(context,self.cfg)

    @torch.inference_mode()
    def predict_encoded(self,contexts,encoded):
        tokens=torch.stack([e[0] for e in encoded]).to(self.device)
        masks=torch.stack([e[1] for e in encoded]).to(self.device)
        se=torch.tensor([c.self_elo for c in contexts],device=self.device)
        oe=torch.tensor([c.oppo_elo for c in contexts],device=self.device)
        logits,_,_=self.model(tokens,se,oe)
        probs=logits.float().masked_fill(~masks,-torch.inf).softmax(-1).cpu()
        result=[]
        for i,(context,(token,mask,board)) in enumerate(zip(contexts,encoded)):
            indices=mask.nonzero().flatten().tolist()
            moves=[self.vocab[j] if board.turn else mirror_move(self.vocab[j]) for j in indices]
            result.append(distribution(board,moves,probs[i,indices].tolist(),'G0',self.record['sha256'],
                                       tensor_hash(token,mask,context.self_elo,context.oppo_elo)))
        return result

    def predict_many(self,contexts):
        return self.predict_encoded(contexts,[self.encode(c) for c in contexts])

    def predict(self,context): return self.predict_many([context])[0]
