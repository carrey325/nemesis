from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import chess
from nemesis.types import PositionContext, MoveDistribution
from nemesis.preflight import verify_file, validate_distribution, require_revision


class TerminalPosition(ValueError):
    """Terminal contexts have no policy and must never enter training."""


def context_from_row(row):
    return PositionContext(row['fen6'], tuple(row['history_uci']),
        row.get('initial_fen', chess.STARTING_FEN), chess.Board(row['fen6']).turn,
        int(row.get('self_elo',1900)), int(row.get('oppo_elo',1900)),
        180, 0, row.get('clock_before'), row.get('history_hash',''))


def rebuild(context):
    board = chess.Board(context.initial_fen)
    if not board.is_valid(): raise ValueError('Invalid initial position')
    for uci in context.history_uci:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves: raise ValueError('Illegal real history: '+uci)
        board.push(move)
    if board.fen() != context.fen6 or board.turn != context.root_color:
        raise ValueError('Historical FEN / mover mismatch')
    if board.is_game_over(): raise TerminalPosition('Terminal position: '+str(board.outcome()))
    return board


def model_record(key, lock_path=Path('artifacts/B00/source_lock.json')):
    lock=json.loads(Path(lock_path).read_text())
    record=lock['models'][key]
    require_revision(record['code_revision'])
    # Verify the code actually imported, not merely the source directory name.
    for source in lock['source_files']:
        if ('maia3-' if key.startswith('maia3') else 'maia2-') in source['path']:
            verify_file(source)
    verify_file(record)
    return record


def tensor_hash(tensor, mask, self_elo, oppo_elo):
    h=hashlib.sha256()
    for t in (tensor, mask): h.update(t.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps([self_elo,oppo_elo]).encode())
    return h.hexdigest()


def lock_fp32():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def distribution(board,moves,probs,model_id,model_hash,input_hash):
    if board.is_game_over(): raise TerminalPosition('No policy for terminal position')
    validate_distribution(board,list(moves),list(probs))
    return MoveDistribution(tuple(moves),tuple(probs),model_id,model_hash,input_hash)
