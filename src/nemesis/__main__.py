import argparse
from pathlib import Path

from .preflight import run_preflight


def main():
    parser = argparse.ArgumentParser(prog="python -m nemesis")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Run bounded B00 runtime checks")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)
    data = commands.add_parser('data', help='B01 public data')
    dc=data.add_subparsers(dest='data_command',required=True)
    dp=dc.add_parser('prepare'); dp.add_argument('--config',type=Path,required=True); dp.add_argument('--out',type=Path,required=True)
    verifier=commands.add_parser('verify'); verifier.add_argument('--artifacts',type=Path,required=True); verifier.add_argument('--out',type=Path,required=True)
    baseline=commands.add_parser('baseline',help='B02 full development-only G0 baseline')
    baseline.add_argument('--config',type=Path,required=True)
    baseline.add_argument('--cohort',choices=['dev','confirm'],required=True)
    baseline.add_argument('--out',type=Path,required=True)
    twins=commands.add_parser('twins',help='B03 frozen-backbone digital twins')
    tc=twins.add_subparsers(dest='twins_command',required=True)
    train=tc.add_parser('train'); train.add_argument('--config',type=Path,required=True); train.add_argument('--out',type=Path,required=True)
    train.add_argument('--resume',action='store_true')
    validate=tc.add_parser('validate'); validate.add_argument('--config',type=Path,required=True); validate.add_argument('--out',type=Path,required=True)
    validate.add_argument('--cohort',choices=['dev','confirm'],required=True); validate.add_argument('--splits',required=True)
    v2=commands.add_parser('v2',help='V2 stage-scoped asset audit')
    vc=v2.add_subparsers(dest='v2_command',required=True)
    audit=vc.add_parser('audit')
    audit.add_argument('--config',type=Path,required=True); audit.add_argument('--registry',type=Path,required=True)
    audit.add_argument('--out',type=Path,required=True)
    screen=vc.add_parser('screen')
    screen.add_argument('--config',type=Path,required=True); screen.add_argument('--assets',type=Path,required=True)
    screen.add_argument('--out',type=Path,required=True)
    utility=vc.add_parser('utility')
    utility.add_argument('--config',type=Path,required=True);utility.add_argument('--assets',type=Path,required=True)
    utility.add_argument('--panels',type=Path,required=True);utility.add_argument('--out',type=Path,required=True)
    utility.add_argument('--resume',action='store_true',help='Explicit verified continuation from the diagnosed B04B selector checkpoint')
    challenge=vc.add_parser('challenge')
    challenge.add_argument('--phase',choices=['lock','train','evaluate'],required=True)
    for flag in ('config','assets','panels','out'):
        challenge.add_argument('--'+flag,type=Path,required=True)
    freeze=vc.add_parser('freeze')
    for flag in ('config','evidence','out'):
        freeze.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.command=='v2':
            if args.v2_command=='freeze':
                from .v2.freeze import freeze_program
                result=freeze_program(args.config,args.evidence,args.out)
                print(f'{result.execution_status}: {result.reason}')
                return result.exit_code
            if args.v2_command=='challenge':
                from .v2.challenge import run_challenge
                result=run_challenge(args.config,args.phase,args.assets,args.panels,args.out)
                print(f'{result.execution_status}: {result.reason}')
                return result.exit_code
            if args.v2_command=='utility':
                if args.resume:
                    from .v2.utility_continue import continue_utility
                    return continue_utility(args.config,args.assets,args.panels,args.out)
                from .v2.utility import run_utility
                result=run_utility(args.config,args.assets,args.panels,args.out)
                print(f'{result.execution_status}: {result.reason}')
                return result.exit_code
            if args.v2_command=='screen':
                from .v2.panels import prepare_panels
                result=prepare_panels(args.config,args.assets,args.out)
                print(f'{result.execution_status}: {result.reason}')
                return result.exit_code
            from .v2.assets import audit_assets
            result=audit_assets(args.config,args.registry,args.out)
            print(f'{result.execution_status}: {result.reason}')
            return result.exit_code
        if args.command=='twins':
            from .models.twins import run_training,run_validation
            if args.twins_command=='train': return run_training(args.config,args.out,args.resume)
            return run_validation(args.config,args.cohort,args.splits,args.out)
        if args.command=='baseline':
            from .evaluation.prediction import run_baseline
            return run_baseline(args.config,args.cohort,args.out)
        if args.command=='data':
            from .data.pipeline import run_data
            return run_data(args.config,args.out)
        if args.command=='verify':
            from .data.pipeline import verify,manifest
            from .data.common import write_json,Meter
            meter=Meter(args.artifacts)
            try: result=verify(args.artifacts,meter)
            finally: meter.checkpoint()
            write_json(args.out,result); manifest(args.artifacts)
            return 0
        result = run_preflight(args.config, args.out)
    except FileExistsError as exc:
        parser.exit(3, str(exc) + "\n")
    except ValueError as exc:
        parser.exit(3,str(exc)+'\n')
    except FileNotFoundError as exc:
        parser.exit(3,str(exc)+'\n')
    except PermissionError as exc:
        parser.exit(4,str(exc)+'\n')
    except RuntimeError as exc:
        parser.exit(5,str(exc)+'\n')
    print(f"{result.execution_status}: {result.reason}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
