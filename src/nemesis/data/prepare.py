"""Strict legal replay and pre-decision clocks. Raw identities stay local."""
import io
import logging
import re
from datetime import datetime,timezone
from pathlib import Path
import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
from .common import digest,stable_hash,write_json
from .splits import split_for

def parse_game(game,source_rated_blitz=False):
    h=game.headers
    reason=None
    if game.errors: reason='illegal_pgn'
    elif h.get('Variant','Standard')!='Standard' or h.get('SetUp')=='1': reason='variant_or_nonstandard_start'
    elif h.get('Event','').casefold()!='rated blitz game' and not source_rated_blitz: reason='not_rated_blitz'
    elif h.get('TimeControl')!='180+0': reason='time_control'
    elif 'BOT' in (h.get('WhiteTitle'),h.get('BlackTitle')): reason='BOT'
    elif h.get('Result') not in {'1-0','0-1','1/2-1/2'}: reason='unfinished'
    try:
        utc=datetime.strptime(h['UTCDate']+' '+h['UTCTime'],'%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc).isoformat()
        elos=[int(h['WhiteElo']),int(h['BlackElo'])]
    except (KeyError,ValueError): utc=None; elos=None; reason=reason or 'missing_date_or_elo'
    board=game.board(); initial=board.fen(); moves=[]; clocks=[]
    for node in game.mainline():
        if node.move not in board.legal_moves: reason='illegal_pgn'; break
        moves.append(node.move.uci()); clocks.append(node.clock()); board.push(node.move)
    site=h.get('Site',''); match=re.fullmatch(r'https://lichess.org/([A-Za-z0-9]{8})(?:/(?:white|black))?',site)
    gid=match.group(1) if match else stable_hash([initial,moves,utc])
    if not moves: reason=reason or 'no_moves'
    if utc and split_for(utc) is None: reason=reason or 'outside_dates'
    return {'game_id':gid,'white':h.get('White','').casefold(),'black':h.get('Black','').casefold(),'utc':utc,'elos':elos,'initial_fen':initial,'moves':moves,'clocks':clocks,'split':split_for(utc) if utc else None,'exclusion':reason}

def input_fingerprint(board, history_boards, self_elo, oppo_elo, ponder):
    # Matches Maia3 dataset.tokenize_board: each historical board mirrors by its
    # own side to move; only piece planes, 8 positions and ponder/100 are tokens.
    positions=[]
    for b in history_boards[-8:]:
        oriented=b.mirror() if b.turn==chess.BLACK else b
        positions.append(oriented.board_fen())
    positions=[positions[0]]*(8-len(positions))+positions
    oriented=board.mirror() if board.turn==chess.BLACK else board
    legal_mask=sorted(move.uci() for move in oriented.legal_moves)
    return stable_hash([positions,self_elo,oppo_elo,ponder,legal_mask])

def decision_rows(g, player, player_id, ply_range=(20,100)):
    if g['exclusion']: return []
    color=player.casefold()==g['white']; board=chess.Board(g['initial_fen'])
    before={True:180.0,False:180.0}; history=[]; boards=[board.copy()]; rows=[]; previous_clock=180.0
    for ply,uci in enumerate(g['moves'],1):
        mover=board.turn; clock=before[mover]; post=g['clocks'][ply-1]
        if mover==color and ply_range[0]<=ply<=ply_range[1]:
            self_elo=g['elos'][0 if color else 1]; opp=g['elos'][1 if color else 0]
            # Missing ponder is retained and excluded, never silently guessed.
            # Pinned UCI _tokens_from_history supplies ponder=0 even when
            # include_time_info=False. Preserve observed clocks for audit, but
            # do not introduce an untrained time feature into the model input.
            ih=input_fingerprint(board,boards,self_elo,opp,0.0)
            rows.append({'sample_id':stable_hash([g['game_id'],player_id,ply]),'game_id':g['game_id'],'player_id':player_id,'split':g['split'],'utc':g['utc'],'ply':ply,'fen6':board.fen(),'initial_fen':g['initial_fen'],'history_uci':list(history),'clock_history':g['clocks'][:ply-1],'mover_color':'white' if mover else 'black','self_elo':self_elo,'oppo_elo':opp,'time_control':'180+0','clock_before':clock,'clock_ponder':0.0,'target_move_uci':uci,'input_hash':ih,'board_hash':stable_hash(board.fen().split()[:4]),'main_eligible':clock is not None and clock>=30,'phase':'opening' if ply<=30 else ('endgame' if len(board.piece_map())<=12 else 'middlegame')})
        # 180+0 clocks must be finite/nonnegative, and cannot increase. Once a
        # clock is corrupt, that color remains unknown for the rest of the game.
        import math
        valid_post=post is not None and math.isfinite(post) and post>=0 and clock is not None and post<=clock
        before[mover]=post if valid_post else None
        previous_clock=post
        board.push_uci(uci); history.append(uci); boards.append(board.copy())
    if len(rows)>20: rows=[rows[round(i*(len(rows)-1)/19)] for i in range(20)]
    return rows

SAMPLE_SCHEMA=pa.schema([(k,t) for k,t in [('sample_id',pa.string()),('game_id',pa.string()),('player_id',pa.string()),('split',pa.string()),('utc',pa.string()),('ply',pa.int32()),('fen6',pa.string()),('initial_fen',pa.string()),('history_uci',pa.list_(pa.string())),('clock_history',pa.list_(pa.float64())),('mover_color',pa.string()),('self_elo',pa.int32()),('oppo_elo',pa.int32()),('time_control',pa.string()),('clock_before',pa.float64()),('clock_ponder',pa.float64()),('target_move_uci',pa.string()),('input_hash',pa.string()),('board_hash',pa.string()),('main_eligible',pa.bool_()),('phase',pa.string())]])

def write_samples(path,rows):
    pq.write_table(pa.Table.from_pylist(rows,schema=SAMPLE_SCHEMA),path,compression='zstd')

def load_games(paths,meter=None,source_rated_blitz=False):
    logger=logging.getLogger('chess.pgn'); old=logger.level; logger.setLevel(logging.CRITICAL)
    try:
        for path in paths:
            with Path(path).open(encoding='utf-8') as f:
                count=0
                while (g:=chess.pgn.read_game(f)) is not None:
                    yield parse_game(g,source_rated_blitz=source_rated_blitz); count+=1
                    if meter and count%500==0: meter.checkpoint()
    finally: logger.setLevel(old)

def coverage(games,player,cohort=None):
    import statistics
    usable=[g for g in games if player in (g['white'],g['black']) and not g['exclusion'] and (not cohort or not (g['white'] in cohort and g['black'] in cohort))]
    fit=[g['elos'][0 if g['white']==player else 1] for g in usable if g['split'].startswith('fit_')]
    median=statistics.median(fit) if fit else None; partitions={}; reasons=[]
    for split,minimum in [('fit_a',300),('fit_b',300),('calibration',100),('H1',150),('H2',150)]:
        bucket=sorted([g for g in usable if g['split']==split],key=lambda g:stable_hash([g['game_id'],20261007]))[:2000]
        n=len(bucket); white=sum(g['white']==player for g in bucket)
        partitions[split]={'games':n,'white':white,'black':n-white,'minimum':minimum}
        if n<minimum: reasons.append(split+'_count')
        if not n or min(white,n-white)/n<0.2: reasons.append(split+'_color')
    if median is None or not 1800<=median<=2000: reasons.append('fit_median_elo')
    if sum(p['games'] for p in partitions.values())<1000: reasons.append('total_count')
    return {'eligible':not reasons,'fit_median_elo':median,'partitions':partitions,'reasons':reasons}

def build_samples(pgn_paths,config,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    import json
    domain='data' in config
    exclusions=json.loads(Path('data/B01/acquisition_checkpoint.json').read_text()).get('source_exclusions',{}) if domain else {}
    lineage=stable_hash([[(Path(p).name,digest(p)) for p in pgn_paths],config,'B01_INPUT_V2_8BOARDS_ELOS_PONDER0_NORMALIZED_LEGAL_MASK',digest('data/B01/candidates_private.json') if domain else None,exclusions])
    receipt=out/'sample_checkpoint.json'; path=out/'samples.parquet'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['lineage']!=lineage or digest(path)!=saved['sha256']: raise ValueError('resume lineage/content mismatch')
        for filename,sha in saved.get('outputs',{}).items():
            if digest(out/filename)!=sha: raise ValueError('resume dataset output mismatch: '+filename)
        from .common import append_json
        append_json(out/'resume_log.jsonl',{'action':'SKIP_IDENTICAL_SAMPLE_PREPARATION','lineage':lineage,'sample_rows':saved['rows']})
        return path
    if domain:
        from .acquire import validate_contract
        from .pipeline import finish_dataset
        from .common import Meter
        validate_contract(config)
        state=json.loads(Path('data/B01/acquisition_checkpoint.json').read_text())
        if state.get('run_id')!='B01-N01': raise ValueError('unrecognized acquisition run ID')
        if state.get('config_hash')!=stable_hash(config): raise ValueError('acquisition config mismatch')
        expected={str(Path(r['path']).resolve()) for r in state['completed']}
        if expected!={str(Path(p).resolve()) for p in pgn_paths}: raise ValueError('source paths differ from acquisition checkpoint')
        rank=json.loads(Path('data/B01/candidates_private.json').read_text())['ranking']
        processed={r['candidate_id'] for r in state['completed']}|set(exclusions)
        if len(processed)<min(128,len(rank)) and state.get('stop_reason')!='32_CROSS_COHORT_VALID': raise ValueError('acquisition incomplete; cannot declare data insufficiency')
        meter=Meter(out)
        try: finish_dataset(config,out,meter)
        finally: meter.checkpoint()
        outputs={p.name:digest(p) for p in out.glob('*.parquet')}
        for name in ['split_manifest.json','leakage_audit.json','candidate_outcomes.json','data_report.md']: outputs[name]=digest(out/name)
        write_json(receipt,{'lineage':lineage,'sha256':digest(path),'rows':pq.read_metadata(path).num_rows,'outputs':outputs,'run_id':state['run_id'],'input_projection':'V2 includes normalized legal move mask'})
        return path
    rows=[]; seen=set()
    for g in load_games(pgn_paths):
        if g['game_id'] in seen: continue
        seen.add(g['game_id'])
        for player,pid in config.get('test_players',{}).items():
            if player in (g['white'],g['black']): rows.extend(decision_rows(g,player,pid,ply_range=tuple(config.get('fixture_ply_range',(20,100)))))
    write_samples(path,rows); write_json(receipt,{'lineage':lineage,'sha256':digest(path),'rows':len(rows)})
    return path
