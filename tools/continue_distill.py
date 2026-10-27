"""Continue full-state distillation while TRAIN FID improves; report validation separately."""
import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_extension(old, new):
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    previous, target = old['settings']['steps'], new['settings']['steps']
    if target <= previous:
        raise ValueError('The new target must exceed the previous target')
    new['settings']['steps'] = previous
    for value in (old, new):
        value.pop('code_sha256', None)
        value.pop('execution', None)
    if old != new:
        raise ValueError('Extension would change the learning, data or TRAIN FID recipe')


def migrate(source, destination, task, receipt):
    import torch
    from fastgen.continuation import fingerprint
    from fastgen.data import read_manifest
    from fastgen.engine import stage_identity
    from fastgen.runtime import atomic_json, atomic_save, digest, sha256
    source, destination = Path(source), Path(destination)
    source_hash = sha256(source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    identity = stage_identity(task, read_manifest(task['manifest'], task['split_dir']))
    check_extension(saved['identity'], identity)
    if saved['signature'] != digest(saved['identity']):
        raise ValueError('Checkpoint signature is inconsistent')
    required = {'model', 'ema', 'optimizer', 'rng', 'batch_state', 'step', 'curve', 'active_seconds'}
    if not required <= saved.keys():
        raise ValueError('A complete training checkpoint is required')
    state = {key: value for key, value in saved.items() if key not in ('identity', 'signature')}
    before = {key: fingerprint(value) for key, value in state.items()}
    old_identity = saved['identity']
    saved['identity'], saved['signature'] = identity, digest(identity)
    atomic_save(destination, saved)
    restored = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
    after = {key: fingerprint(restored[key]) for key in state}
    if before != after:
        raise ValueError('Full-state preservation check failed')
    if source.resolve() != destination.resolve() and sha256(source) != source_hash:
        raise ValueError('The source checkpoint changed')
    atomic_json(receipt, {'source': str(source), 'source_sha256': source_hash,
        'destination': str(destination), 'step': saved['step'], 'previous_identity': old_identity,
        'identity': identity, 'full_state_equal': True, 'learning_updates_during_migration': 0,
        'fingerprints_before': before, 'fingerprints_after': after})
    atomic_json(destination.parent / 'identity.json', identity)


def copy_source(directory, inventory):
    for name in inventory:
        target = directory / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def validation_report(directory, source, task, checkpoint, keep_previous):
    """This result is never read by the continuation or checkpoint-selection rule."""
    from fastgen.runtime import atomic_json, sha256
    from fastgen.submission import export_ema, official_evaluation
    import torch
    step = torch.load(checkpoint, map_location='meta', weights_only=False)['step']
    report = directory / 'validation_reports' / f'step_{step:07d}_{time.time_ns()}'
    submission = report / 'submission'
    inventory = json.loads((directory / 'pipeline_identity.json').read_text())['code']
    for name in inventory:
        target = submission / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / 'source_snapshot' / name, target)
    split = submission / 'data/pokemon-generation-one-22k'
    split.mkdir(parents=True, exist_ok=True)
    for name in ('train_split.txt', 'val_split.txt', 'category_to_id.json'):
        shutil.copy2(Path(task['split_dir']) / name, split / name)
    checkpoints = submission / 'checkpoints'
    checkpoints.mkdir(parents=True)
    shutil.copy2(source / 'submission/checkpoints/few_nfe.ckpt', checkpoints / 'few_nfe.ckpt')
    export_ema(checkpoint, checkpoints / 'one_nfe.ckpt', 'one_nfe')
    job = {'mode': 'one_nfe', 'run_root': str(report), 'data_root': task['data_root'],
           'split_dir': task['split_dir'], 'device': task['device'], 'batch_size': task['eval_batch_size'],
           'seed': 20261002, 'checkpoint_sha256': sha256(checkpoints / 'one_nfe.ckpt')}
    atomic_json(directory / 'status.json', {'phase': 'running', 'stage': 'official_one_nfe', 'unix': time.time()})
    official_evaluation(job)
    result = json.loads((report / 'official_evaluation/one_nfe.json').read_text())
    result.update(step=step, checkpoint_source=str(checkpoint), report_only=True,
                  used_for_training_or_selection=False)
    atomic_json(directory / 'latest_validation.json', result)
    # Preserve scores, logs and images; keep only the latest report's exported weights.
    if not keep_previous:
        for previous in (directory / 'validation_reports').glob('step_*/submission/checkpoints'):
            if previous.resolve() == checkpoints.resolve():
                continue
            for name in ('one_nfe.ckpt', 'few_nfe.ckpt'):
                path = previous / name
                if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
                    path.unlink()
    return result


def run(args):
    import torch
    from fastgen.engine import train_stage, learning_rate
    from fastgen.project import check_official, code_inventory, run_lock
    from fastgen.runtime import append_event, atomic_json, digest, sha256
    from fastgen.submission import export_ema, bundle_submission, write_facts, evaluate_submission
    source, directory = args.source_run.resolve(strict=True), args.run_root.resolve()
    if source == directory or source in directory.parents or directory in source.parents:
        raise ValueError('Use a separate sibling output directory')
    if args.block_steps < 5000 or args.block_steps % 5000 or args.patience < 1:
        raise ValueError('Use blocks divisible by 5000 and positive patience')
    check_official()
    old_identity = json.loads((source / 'pipeline_identity.json').read_text())
    # Existing runtime files must match the actual completed run; the controller is the only addition.
    for name, expected in old_identity['code'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Original training source changed: ' + name)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        started = time.perf_counter()
        control_path = directory / 'continuation_control.json'
        if control_path.exists():
            control = json.loads(control_path.read_text())
            if control['source_run'] != str(source) or control['block_steps'] != args.block_steps or control['patience'] != args.patience:
                raise ValueError('Continuation control changed')
            if control['phase'] == 'complete':
                return
            task = json.loads((directory / 'tasks/distill.json').read_text())
        else:
            launcher_files = {'.pipeline.lock', 'launch_tmux.sh', 'launch.json', 'console.log'}
            if any(p.name not in launcher_files for p in directory.iterdir()):
                raise ValueError('The destination must be fresh')
            for name in ('teacher', 'pilot_flip', 'pilot_spatial', 'pairs_teacher', 'data'):
                (directory / name).symlink_to(source / name, target_is_directory=True)
            (directory / 'distill').mkdir()
            (directory / 'tasks').mkdir()
            (directory / 'adopted_history').mkdir()
            for name in ('execution.jsonl', 'augmentation_selection.json', 'preflight.json'):
                if (source / name).exists():
                    shutil.copy2(source / name, directory / name)
            for name in ('events.jsonl', 'metrics.jsonl', 'environment.json', 'summary.json'):
                shutil.copy2(source / 'distill' / name, directory / 'distill' / name)
            for name in ('samples', 'sessions'):
                shutil.copytree(source / 'distill' / name, directory / 'distill' / name)
            inventory = code_inventory()
            copy_source(directory, inventory)
            task = json.loads((source / 'tasks/distill.json').read_text())
            saved = torch.load(source / 'distill/current.pt', map_location='meta', weights_only=False)
            if saved['step'] != task['settings']['steps']:
                raise ValueError('Initial extension expects a completed source stage')
            previous_best = saved['best_fid']
            task['settings']['steps'] = saved['step'] + args.block_steps
            task['run_dir'] = str(directory / 'distill')
            task['minimum_free_bytes'] = 4 * 1024**3
            task['code_sha256'] = digest(inventory)
            config = copy.deepcopy(old_identity['config'])
            config['distill']['steps'] = task['settings']['steps']
            config['minimum_free_bytes'] = task['minimum_free_bytes']
            atomic_json(directory / 'pipeline_identity.json', {**old_identity, 'code': inventory, 'config': config})
            for name in ('current.pt', 'best.pt'):
                migrate(source / 'distill' / name, directory / 'distill' / name, task,
                        directory / 'adopted_history' / ('initial_' + name + '.json'))
            atomic_json(directory / 'tasks/distill.json', task)
            control = {'source_run': str(source), 'block_steps': args.block_steps, 'patience': args.patience,
                       'phase': 'running', 'completed_blocks': 0, 'stale_blocks': 0,
                       'previous_best_train_fid': previous_best, 'stop_requested': False,
                       'selection_reference': 'TRAIN755 only', 'validation_role': 'report only',
                       'learning_rate_horizon_unchanged': task['settings']['horizon']}
            atomic_json(control_path, control)
        append_event(directory / 'execution.jsonl', event='pipeline_session_start')
        try:
            while True:
                atomic_json(directory / 'status.json', {'phase': 'running', 'stage': 'distill', 'unix': time.time()})
                began = time.perf_counter()
                append_event(directory / 'execution.jsonl', event='child_start', name='distill', action='full_state_extension')
                summary = train_stage(task)
                append_event(directory / 'execution.jsonl', event='child_end', name='distill', action='full_state_extension',
                             returncode=0, process_wall_seconds=time.perf_counter()-began)
                if summary['phase'] != 'complete':
                    control['phase'] = 'stopped'
                    atomic_json(control_path, control)
                    atomic_json(directory / 'status.json', {'phase': 'stopped', 'stage': 'distill', 'unix': time.time()})
                    return
                best = summary['best_train_proxy_fid']
                if control.get('last_decision_step') != summary['step']:
                    improved = best < control['previous_best_train_fid']
                    control['stale_blocks'] = 0 if improved else control['stale_blocks'] + 1
                    control['previous_best_train_fid'] = best
                    control['completed_blocks'] += 1
                    # Decide from TRAIN before opening validation, so validation cannot control continuation.
                    control['continue_from_train'] = control['stale_blocks'] < args.patience
                    control['last_decision_step'] = summary['step']
                    atomic_json(control_path, control)
                    append_event(directory / 'execution.jsonl', event='continuation_decision', step=summary['step'],
                                 best_train_fid=best, improved=improved, continue_training=control['continue_from_train'],
                                 validation_used=False)
                gc.collect()
                torch.cuda.empty_cache()
                result = validation_report(directory, source, task, directory / 'distill/current.pt', args.keep_report_checkpoints)
                print(json.dumps({'step': summary['step'], 'best_TRAIN755_FID': best,
                                  'validation3020_FID': result['FID_printed_by_official_evaluator'],
                                  'continue_from_TRAIN_only': control['continue_from_train']}), flush=True)
                if not control['continue_from_train']:
                    break
                task['settings']['steps'] += args.block_steps
                for name in ('current.pt', 'best.pt'):
                    migrate(directory / 'distill' / name, directory / 'distill' / name, task,
                            directory / 'adopted_history' / f'extend_{task["settings"]["steps"]}_{name}.json')
                atomic_json(directory / 'tasks/distill.json', task)
            checkpoints = directory / 'submission/checkpoints'
            checkpoints.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'submission/checkpoints/few_nfe.ckpt', checkpoints / 'few_nfe.ckpt')
            export_ema(directory / 'distill/best.pt', checkpoints / 'one_nfe.ckpt', 'one_nfe')
            control['phase'] = 'complete'
            atomic_json(control_path, control)
            bundle_submission(directory)
            config = json.loads((directory / 'pipeline_identity.json').read_text())['config']
            evaluate_submission(directory, Path(task['data_root']), Path(task['split_dir']), config)
            shutil.copy2(control_path, directory / 'submission/records/continuation_control.json')
            for report in (directory / 'validation_reports').glob('step_*/official_evaluation/one_nfe.json'):
                target = directory / 'submission/records/validation_reports' / report.parent.parent.name / 'one_nfe.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report, target)
            bundle_submission(directory)
            atomic_json(directory / 'status.json', {'phase': 'complete', 'unix': time.time()})
        except BaseException as error:
            atomic_json(directory / 'status.json', {'phase': 'failed', 'error': repr(error), 'unix': time.time()})
            raise
        finally:
            append_event(directory / 'execution.jsonl', event='pipeline_session_end', session_wall_seconds=time.perf_counter()-started)
            write_facts(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--block-steps', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--keep-report-checkpoints', action='store_true')
    run(parser.parse_args())
