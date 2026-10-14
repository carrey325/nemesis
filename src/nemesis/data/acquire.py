"""Serial official export, bounded header-only monthly discovery."""
import collections
import hashlib
import json
import re
import time
from datetime import datetime,timezone
from pathlib import Path
import requests
import pyzstd
from .common import digest,stable_hash,write_json,append_json

MONTH_URL='https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst'
class SourceBlocked(RuntimeError): pass

def recover_disabled_exclusion(local,out,state,mapping):
    """Replay only the three separately authorized, evidence-bound cases offline."""
    if set(state.get('source_exclusions',{}))-{'C014','C060','C062'}:
        raise ValueError('C014 only under Q01; C060/C062 only under their specific Router interpretations')
    _recover_c014_exclusion(local,out,state,mapping)
    _recover_named_exclusion(local,out,state,mapping,'C060','N05_C060_metadata_diagnostic.json','N05_source_blocker_receipt.json')
    _recover_named_exclusion(local,out,state,mapping,'C062','N05_C062_metadata_diagnostic.json','N05b_source_blocker_receipt.json')


def _recover_c014_exclusion(local,out,state,mapping):
    """Preserve the original C014-only Q01 receipt and persisted evidence."""
    from urllib.parse import urlparse,parse_qs
    receipt_path=out/'account_availability_receipt.json'
    if not receipt_path.exists():
        if 'C014' in state.get('source_exclusions',{}): raise ValueError('missing source-exclusion evidence')
        return
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    export=receipt['original_export_receipt']
    checks=[r for r in receipt['checks'] if r['candidate_id']=='C014']
    if export['candidate_id']!='C014' or export['status']!=404 or export['window']!='fit' or len(checks)!=1 or checks[0]['status']!=200 or checks[0].get('disabled') is not True:
        raise ValueError('unapproved source-exclusion evidence')
    meta_path=local/'C014_account_check.json'
    if digest(meta_path)!=checks[0]['body_sha256']: raise ValueError('account metadata hash mismatch')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('disabled') is not True or meta.get('username','').casefold()!=mapping['C014'].casefold(): raise ValueError('disabled identity evidence mismatch')
    lines=(local/'requests.jsonl').read_text(encoding='utf-8').splitlines()
    record=json.loads(lines[export['line']-1]); url=urlparse(record['url']); query=parse_qs(url.query)
    fit=export_windows()[0][1]
    if record.get('status')!=404 or record.get('utc')!=export['utc'] or url.scheme!='https' or url.netloc!='lichess.org' or url.path!='/api/games/user/'+mapping['C014'] or any(query.get(k)!=[str(fit[k])] for k in ['since','until','max']):
        raise ValueError('historical export evidence mismatch')
    evidence={'reason':'SOURCE_UNAVAILABLE','terminal':True,'interpretation':'lw-nemesis-stage1-B01-Q01; C014 only','candidate_id':'C014','export_status':404,'metadata_status':200,'disabled':True,'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),'metadata_path':str(meta_path),'metadata_sha256':digest(meta_path),'request_line':export['line'],'request_record_sha256':stable_hash(record),'original_requests_sha256':export['sha256']}
    previous=state.get('source_exclusions',{}).get('C014')
    if previous is not None and previous!=evidence: raise ValueError('terminal source exclusion changed')
    if previous is None:
        if digest(local/'requests.jsonl')!=export['sha256']: raise ValueError('original requests receipt hash mismatch')
        if any(r['candidate_id']=='C014' for r in state['completed']): raise ValueError('excluded candidate already completed')
        state.setdefault('source_exclusions',{})['C014']=evidence
        write_json(local/'acquisition_checkpoint.json',state)
        append_json(out/'resume_log.jsonl',{'action':'TERMINAL_SOURCE_EXCLUSION_Q01','candidate_id':'C014','evidence_sha256':stable_hash(evidence),'run_id':state['run_id'],'worker_attempt_id':'B01-N02','export_retried':False})


def _recover_named_exclusion(local,out,state,mapping,cid,diagnostic_filename,blocker_filename):
    """Fixed C060/C062 call sites only; no network or general disabled-account rule."""
    from urllib.parse import urlparse,parse_qs
    authorization_path=out/(cid+'_source_interpretation.json')
    previous=state.get('source_exclusions',{}).get(cid)
    if not authorization_path.exists():
        if previous is not None: raise ValueError(f'missing {cid} authorization')
        return
    authorization=json.loads(authorization_path.read_text(encoding='utf-8'))
    action=f'lw-nemesis-stage1-B01-N05-{cid}-resume'
    if authorization.get('candidate_id')!=cid or authorization.get('action_id')!=action or cid not in mapping:
        raise ValueError(f'unapproved {cid} source interpretation')
    diagnostic_path=out/diagnostic_filename; blocker_path=out/blocker_filename
    if digest(diagnostic_path)!=authorization['diagnostic_receipt_sha256'] or digest(blocker_path)!=authorization['blocker_receipt_sha256']:
        raise ValueError(f'{cid} immutable receipt hash mismatch')
    diagnostic=json.loads(diagnostic_path.read_text(encoding='utf-8')); blocker=json.loads(blocker_path.read_text(encoding='utf-8'))
    if diagnostic.get('candidate_id')!=cid or diagnostic.get('status')!=200 or diagnostic.get('disabled') is not True or diagnostic.get('disabled_field_present') is not True or diagnostic.get('identity_binding_verified') is not True or diagnostic.get('metadata_request_count')!=1 or diagnostic.get('retries')!=0:
        raise ValueError(f'unapproved {cid} metadata evidence')
    if blocker.get('candidate_id',blocker.get('blocked_candidate'))!=cid or blocker.get('window')!='fit' or blocker.get('http_status')!=404 or diagnostic['original_export_record_sha256']!=blocker['request_record_sha256'] or diagnostic['original_requests_sha256']!=blocker['raw_requests_sha256']:
        raise ValueError(f'{cid} export receipt mismatch')
    meta_path=local/(cid+'_account_check.json'); private_path=local/(cid+'_metadata_request.json')
    if digest(meta_path)!=diagnostic['body_sha256'] or digest(private_path)!=diagnostic['raw_metadata_request_receipt_sha256']:
        raise ValueError(f'{cid} metadata hash mismatch')
    meta=json.loads(meta_path.read_text(encoding='utf-8')); private=json.loads(private_path.read_text(encoding='utf-8')); name=mapping[cid].casefold()
    if meta.get('disabled') is not True or meta.get('username','').casefold()!=name or meta.get('id','').casefold()!=name:
        raise ValueError(f'{cid} disabled identity mismatch')
    metadata_url=urlparse(private.get('url',''))
    if private.get('candidate_id')!=cid or private.get('status')!=200 or private.get('attempt')!=1 or metadata_url.scheme!='https' or metadata_url.netloc!='lichess.org' or metadata_url.path!='/api/user/'+mapping[cid] or metadata_url.query:
        raise ValueError(f'{cid} private metadata request mismatch')
    records=[json.loads(line) for line in (local/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    record=records[blocker['request_line']-1]; url=urlparse(record.get('url','')); query=parse_qs(url.query); fit=export_windows()[0][1]
    if record.get('status')!=404 or record.get('utc')!=blocker['request_utc'] or stable_hash(record)!=blocker['request_record_sha256'] or url.scheme!='https' or url.netloc!='lichess.org' or url.path!='/api/games/user/'+mapping[cid] or any(query.get(k)!=[str(fit[k])] for k in ['since','until','max']):
        raise ValueError(f'{cid} historical export evidence mismatch')
    if sum(urlparse(r.get('url','')).path==url.path for r in records)!=1:
        raise ValueError(f'{cid} export must remain a single saved request')
    evidence={'reason':'SOURCE_UNAVAILABLE','terminal':True,'interpretation':action,'candidate_id':cid,'export_status':404,'metadata_status':200,'disabled':True,'receipt_path':str(diagnostic_path),'receipt_sha256':digest(diagnostic_path),'blocker_receipt_path':str(blocker_path),'blocker_receipt_sha256':digest(blocker_path),'authorization_path':str(authorization_path),'authorization_sha256':digest(authorization_path),'metadata_path':str(meta_path),'metadata_sha256':digest(meta_path),'private_metadata_receipt_sha256':digest(private_path),'request_line':blocker['request_line'],'request_record_sha256':stable_hash(record),'original_requests_sha256':blocker['raw_requests_sha256']}
    if previous is not None and previous!=evidence: raise ValueError(f'{cid} terminal source exclusion changed')
    if previous is None:
        if digest(local/'requests.jsonl')!=blocker['raw_requests_sha256']: raise ValueError(f'{cid} original requests receipt hash mismatch')
        if any(r['candidate_id']==cid for r in state['completed']): raise ValueError(f'{cid} excluded candidate already completed')
        state.setdefault('source_exclusions',{})[cid]=evidence
        write_json(local/'acquisition_checkpoint.json',state)
        append_json(out/'resume_log.jsonl',{'action':f'TERMINAL_SOURCE_EXCLUSION_ROUTER_{cid}','candidate_id':cid,'evidence_sha256':stable_hash(evidence),'run_id':state['run_id'],'worker_attempt_id':'B01-N05','export_retried':False})

def stream_chunks(response,deadline_seconds=1200):
    """read1 returns available bytes, so keepalives cannot hide the wall limit."""
    import urllib3
    import threading
    import socket
    expired=threading.Event()
    def abort():
        expired.set()
        try: response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except (AttributeError,OSError):
            try: response.close()
            except (AttributeError,OSError): pass
    timer=threading.Timer(deadline_seconds,abort); timer.daemon=True; timer.start()
    try:
        while True:
            chunk=response.raw.read1(65536,decode_content=True)
            if expired.is_set(): raise requests.Timeout('HTTP body deadline exceeded')
            if not chunk: return
            yield chunk
    except (urllib3.exceptions.HTTPError,OSError) as exc:
        if expired.is_set(): raise requests.Timeout('HTTP body deadline exceeded') from exc
        raise requests.ConnectionError('HTTP body read failed') from exc
    finally: timer.cancel()

def retry_parameters(params,failures):
    result=dict(params)
    if failures: result['sort']='dateDesc'
    return result

def validate_contract(config):
    expected={'timezone':'UTC','variant':'standard','rated':True,'time_control':'180+0','fit_median_elo_range':[1800,2000],'cohort_seed':20261007,'dev_players':8,'confirm_players':24,'candidate_month':'2026-03','candidate_game_scan_cap':1000000,'candidate_compressed_gib_cap':2,'history_candidates_cap':128,'history_games_per_candidate_cap':10000,'minimum_games':{'fit_a':300,'fit_b':300,'calibration':100,'H1':150,'H2':150},'maximum_games_per_partition':2000,'minimum_color_fraction':0.2,'clock_before_min_seconds':30,'require_known_clock':True,'training_ply_range':[20,100],'max_decisions_per_game':20,'intervals':{'fit':['2025-08-01','2026-04-01'],'calibration':['2026-04-01','2026-05-01'],'H1':['2026-05-01','2026-08-01'],'H2':['2026-08-01','2026-10-01']},'exclude_cohort_vs_cohort_games':True,'no_performance_based_player_replacement':True}
    for key,value in expected.items():
        if config.get('data',{}).get(key)!=value: raise ValueError('Unapproved B01 contract value: '+key)

def export_parameters():
    return {'since':1754006400000,'until':1790812800000-1,'max':10000,'rated':'true','perfType':'blitz','moves':'true','tags':'true','clocks':'true','evals':'false','opening':'false','ongoing':'false','finished':'true','sort':'dateAsc'}

def export_windows():
    # Dates are the approved global intervals. Quotas share the single 10,000
    # game budget; no window or time control is broadened.
    result=[]
    for window,start,end,cap in [('fit','2025-08-01','2026-04-01',4000),('calibration','2026-04-01','2026-05-01',500),('H1','2026-05-01','2026-08-01',500),('H2','2026-08-01','2026-10-01',500)]:
        params=export_parameters(); params.update(since=int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()*1000),until=int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()*1000)-1,max=cap)
        result.append((window,params))
    return result

def discovery_headers(lines,cap=1000000):
    header={}; in_moves=False; count=0
    for line in lines:
        match=re.fullmatch(r'\[(\w+) "(.*)"\]\s*',line)
        if match:
            if in_moves:
                yield header; count+=1
                if count>=cap: return
                header={}; in_moves=False
            # Discovery never reads Result, comments, analysis or move content.
            if match[1] in {'Event','TimeControl','UTCDate','White','Black','WhiteElo','BlackElo','WhiteTitle','BlackTitle','Variant'}: header[match[1]]=match[2]
        elif line.strip(): in_moves=True
    if header and in_moves and count<cap: yield header

def rank_candidates(headers):
    counts=collections.Counter()
    for h in headers:
        if h.get('Event')!='Rated Blitz game' or h.get('TimeControl')!='180+0' or h.get('Variant','Standard')!='Standard' or not h.get('UTCDate','').startswith('2026.03.'): continue
        if 'BOT' in (h.get('WhiteTitle'),h.get('BlackTitle')): continue
        for color in ['White','Black']:
            try: elo=int(h[color+'Elo'])
            except (KeyError,ValueError): continue
            if 1800<=elo<=2000 and h.get(color+'Title')!='BOT' and h.get(color): counts[h[color].casefold()]+=1
    return sorted(counts.items(),key=lambda r:(-r[1],r[0]))

def request_stream(url,params,receipt_path,single=False):
    for attempt in range(1,4):
        started=datetime.now(timezone.utc).isoformat()
        try:
            r=requests.get(url,params=params,headers={'Accept':'application/x-chess-pgn' if params else 'application/octet-stream','Accept-Encoding':'identity','User-Agent':'Nemesis-public-research-B01/1.0'},stream=True,timeout=(30,90))
            record={'url':r.url,'utc':started,'attempt':attempt,'status':r.status_code,'retry_after':r.headers.get('Retry-After'),'content_type':r.headers.get('Content-Type'),'content_encoding':r.headers.get('Content-Encoding'),'transfer_encoding':r.headers.get('Transfer-Encoding')}
            append_json(receipt_path,record)
            if r.status_code in (401,403,404): r.close(); raise SourceBlocked('Official source unavailable/refused: HTTP '+str(record['status']))
            if r.status_code==429 or r.status_code>=500:
                delay=max(60,float(r.headers.get('Retry-After','60'))); r.close()
                if single:
                    exc=requests.HTTPError('retryable HTTP '+str(record['status']))
                    exc.retry_delay=delay
                    raise exc
                if attempt==3: raise SourceBlocked('Persistent official rate/source failure')
                time.sleep(delay); continue
            r.raise_for_status(); return r
        except requests.RequestException as exc:
            append_json(receipt_path,{'utc':started,'attempt':attempt,'failure':type(exc).__name__})
            if single: raise
            if attempt==3: raise SourceBlocked('Persistent official network failure') from exc
            time.sleep(60)
    raise SourceBlocked('request exhausted')

def scan_month(local,out,meter=None):
    local.mkdir(parents=True,exist_ok=True)
    private=local/'candidates_private.json'
    if private.exists():
        saved=json.loads(private.read_text())
        if digest(local/'month_prefix.pgn.zst')!=saved['prefix_sha256']: raise ValueError('month checkpoint hash mismatch')
        return saved
    response=request_stream(MONTH_URL,None,local/'requests.jsonl')
    compressed=local/'month_prefix.pgn.zst'; dec=pyzstd.EndlessZstdDecompressor(); read_bytes=0; pending=b''; count=0; counts=collections.Counter(); h={}; in_moves=False; stopped=False
    keys={'Event','TimeControl','UTCDate','White','Black','WhiteElo','BlackElo','WhiteTitle','BlackTitle','Variant'}
    def finish():
        for name,n in rank_candidates([h]): counts[name]+=n
    try:
        with compressed.open('wb') as f:
            while read_bytes<2*1024**3 and count<1000000:
                chunk=response.raw.read(min(65536,2*1024**3-read_bytes))
                if not chunk: break
                f.write(chunk); read_bytes+=len(chunk); pending+=dec.decompress(chunk)
                lines=pending.split(b'\n'); pending=lines.pop()
                for raw in lines:
                    line=raw.decode('utf-8'); match=re.fullmatch(r'\[(\w+) "(.*)"\]\s*',line)
                    if match:
                        if in_moves:
                            finish(); count+=1; h={}; in_moves=False
                            if count>=1000000: stopped=True; break
                        if match[1] in keys: h[match[1]]=match[2]
                    elif line.strip(): in_moves=True
                if stopped: break
                if count and count%10000<100 and meter: meter.checkpoint()
    finally: response.close()
    if count<1000000 and in_moves and dec.at_frame_edge: finish(); count+=1
    ordered=sorted(counts.items(),key=lambda r:(-r[1],r[0]))
    saved={'url':MONTH_URL,'scan_games':count,'compressed_bytes':read_bytes,'stop_reason':'GAME_CAP' if count==1000000 else ('BYTE_CAP' if read_bytes==2*1024**3 else 'EOF'),'prefix_sha256':digest(compressed),'ranking':ordered,'frozen_utc':datetime.now(timezone.utc).isoformat(),'ranking_rule':'qualifying occurrences descending, casefold username ascending; no labels or performance'}
    write_json(private,saved)
    public={k:v for k,v in saved.items() if k!='ranking'}
    public['candidate_order']=[{'candidate_id':f'C{i:03d}','occurrences':n} for i,(_,n) in enumerate(ordered,1)]
    public['private_order_sha256']=digest(private); write_json(out/'candidates_manifest.json',public)
    return saved

def acquire_games(config,out_dir):
    validate_contract(config)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); local=Path('data/B01'); local.mkdir(parents=True,exist_ok=True)
    from .common import Meter
    meter=Meter(out); config_hash=stable_hash(config); checkpoint=local/'acquisition_checkpoint.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text())['config_hash']!=config_hash: raise ValueError('acquisition config changed')
    state={'config_hash':config_hash,'run_id':'B01-N01','completed':[]} if not checkpoint.exists() else json.loads(checkpoint.read_text())
    policy={'windows':export_windows(),'games_requested_per_user':5500,'per_user_total_cap':10000,'fixed_before_any_completed_candidate':True,'reason':'disjoint approved date windows preserve time coverage within D1 maximum; no performance observed; incomplete earlier C001 attempts retained and excluded from the dataset','timeout_repair':'hard 1200-second body deadline; identity encoding; documented dateDesc fallback after a stream/network failure; same endpoint, range and cap; added before any cohort/model results'}
    policy_path=out/'history_export_policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text())!=json.loads(json.dumps(policy)): raise ValueError('history export policy changed during resume')
    write_json(policy_path,policy)
    state.setdefault('windows',{}); state.setdefault('retry_failures',{})
    write_json(checkpoint,state)
    try:
        discovery=scan_month(local,out,meter)
        from .prepare import load_games,coverage
        mapping={f'C{j:03d}':user for j,(user,_) in enumerate(discovery['ranking'][:128],1)}
        recover_disabled_exclusion(local,out,state,mapping)
        for prior in state['completed']:
            if digest(Path(prior['path']))!=prior['sha256']: raise ValueError('history resume hash mismatch')
            if prior.get('coverage_schema_version')!=2:
                old=prior['coverage']; unique={g['game_id']:g for g in load_games([Path(prior['path'])],meter,source_rated_blitz=True)}
                prior['coverage']=coverage(list(unique.values()),mapping[prior['candidate_id']]); prior['coverage_schema_version']=2
                append_json(out/'resume_log.jsonl',{'action':'REQUALIFY_UNCHANGED_SOURCE_EVENT_FORMAT_FIX','candidate_id':prior['candidate_id'],'old_eligible':old['eligible'],'new_eligible':prior['coverage']['eligible'],'source_sha256':prior['sha256']})
        write_json(checkpoint,state)
        if state.get('stop_reason')=='32_CROSS_COHORT_VALID':
            for prior in state['completed']:
                if digest(Path(prior['path']))!=prior['sha256']: raise ValueError('history resume hash mismatch')
            append_json(out/'resume_log.jsonl',{'action':'SKIP_FROZEN_COMPLETED_ACQUISITION','completed_candidates':len(state['completed']),'config_hash':config_hash,'run_id':state['run_id']})
            return checkpoint
        for i,(name,n) in enumerate(discovery['ranking'][:128],1):
            cid=f'C{i:03d}'; final=local/(cid+'.pgn')
            if cid in state.get('source_exclusions',{}): continue
            previous=next((r for r in state['completed'] if r['candidate_id']==cid),None)
            if previous:
                if digest(final)!=previous['sha256']: raise ValueError('history resume hash mismatch')
                continue
            pause=local/'pause_after_candidate.txt'
            if pause.exists() and pause.read_text(encoding='utf-8').strip() in {r['candidate_id'] for r in state['completed']}:
                from .common import ResourceLimit
                raise ResourceLimit('B01 continuation pause at candidate boundary; remove local pause marker to resume')
            windows=export_windows()
            sources=[]
            for window,params in windows:
                window_path=local/(cid+'_'+window+'.pgn'); key=cid+'/'+window
                if key in state['windows']:
                    if digest(window_path)!=state['windows'][key]['sha256']: raise ValueError('window resume hash mismatch')
                    sources.append(window_path); continue
                temp=window_path.with_suffix('.partial')
                while True:
                    meter.checkpoint(); response=None; started=time.monotonic(); total=0
                    try:
                        actual_params=retry_parameters(params,state['retry_failures'].get(cid,0))
                        response=request_stream('https://lichess.org/api/games/user/'+name,actual_params,local/'requests.jsonl',single=True)
                        with temp.open('wb') as f:
                            for chunk in stream_chunks(response,deadline_seconds=max(1,1200-(time.monotonic()-started))):
                                if chunk: f.write(chunk); f.flush(); total+=len(chunk)
                                if time.monotonic()-started>1200: raise requests.Timeout('export exceeded bounded 20-minute stream window')
                                if total>100*1024**2: raise SourceBlocked('unexpected per-user response exceeds 100 MiB safety bound')
                        break
                    except requests.RequestException as exc:
                        state['retry_failures'][cid]=state['retry_failures'].get(cid,0)+1
                        write_json(checkpoint,state)
                        append_json(local/'requests.jsonl',{'candidate_id':cid,'window':window,'failure_number':state['retry_failures'][cid],'failure':type(exc).__name__,'partial_bytes':temp.stat().st_size if temp.exists() else None,'partial_sha256':digest(temp) if temp.exists() else None})
                        if state['retry_failures'][cid]>=3: raise SourceBlocked('stream/request failed three times for candidate; partial and checkpoints retained locally') from exc
                        time.sleep(max(60,getattr(exc,'retry_delay',60)))
                    finally:
                        if response is not None: response.close()
                temp.replace(window_path); state['windows'][key]={'sha256':digest(window_path),'bytes':total,'max_games':params['max']}; write_json(checkpoint,state); sources.append(window_path)
                print(json.dumps({'candidate_id':cid,'completed_window':window,'bytes':total}),flush=True)
            with final.open('wb') as f:
                for source in sources:
                    with source.open('rb') as sf:
                        for chunk in iter(lambda:sf.read(65536),b''): f.write(chunk)
                    f.write(b'\n\n')
            total=final.stat().st_size
            row={'candidate_id':cid,'sha256':digest(final),'bytes':total,'path':str(final),'utc':datetime.now(timezone.utc).isoformat()}; state['completed'].append(row)
            from .prepare import load_games,coverage
            games=list(load_games([final],meter,source_rated_blitz=True)); seen={}; unique=[]
            for g in games:
                if g['game_id'] not in seen: unique.append(g); seen[g['game_id']]=True
            row['coverage']=coverage(unique,name)
            row['coverage_schema_version']=2
            row['downloaded_games']=len(games); row['unique_games']=len(unique)
            if len(games)>10000: raise ValueError('history game cap violated')
            write_json(checkpoint,state); meter.checkpoint()
            print(json.dumps({'completed_candidates':i,'candidate_id':cid,'bytes':total,'coverage_eligible':row['coverage']['eligible']}),flush=True)
            eligible=[r for r in state['completed'] if r['coverage']['eligible']]
            if len(eligible)>=32:
                mapping={f'C{j:03d}':user for j,(user,_) in enumerate(discovery['ranking'][:128],1)}
                chosen={mapping[r['candidate_id']] for r in eligible[:32]}
                valid=True
                for r in eligible[:32]:
                    c=coverage(list(load_games([Path(r['path'])],meter,source_rated_blitz=True)),mapping[r['candidate_id']],chosen)
                    if not c['eligible']: valid=False; break
                if valid: state['stop_reason']='32_CROSS_COHORT_VALID'; write_json(checkpoint,state); break
        return checkpoint
    finally: meter.checkpoint()
