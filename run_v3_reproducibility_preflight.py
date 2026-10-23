"""Read-only, hash-bound v3 experiment preflight. No model execution or training."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid
import yaml


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def child(root, name):
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its supplied directory: ' + name)
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir')
    for name in ('input_dir', 'comparison_dir', 'residual_dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--contract', type=Path, default=Path(__file__).with_name('v3_experiment_contract.json'))
    args = p.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    processed = Path(get_processed_folder(config))
    out = processed / 'v2_validation' / 'v3_reproducibility_preflight' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='BLOCKED', scope='saved artifact and source contract; no numerical replay',
                  model_forwards=0, backward_calls=0, optimizer_steps=0, checks=[], warnings=[],
                  configuration=config, python=platform.python_version(), runtime={},
                  scientific_validation_passed=False, numerical_reproduction='NOT_TESTED',
                  methodology_changes_permitted=True,
                  research_goal='Useful representations and discrimination between user behaviors; forecasting is a candidate learning mechanism, not sufficient evidence.',
                  legacy_training_entrypoint_supported=False)
    tracked = {}
    def check(name, condition, detail=None):
        report['checks'].append(dict(name=name, passed=bool(condition), detail=detail))
    def verify(path, expected, label, text_source=False):
        try:
            actual = sha256(path)
            tracked[str(path.resolve())] = actual
            normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if text_source else None
            # Historical reports may hash CRLF while a checkout uses LF (or vice versa).
            crlf = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest() if text_source else None
            matched = actual == expected or (text_source and expected in (normalized, crlf))
            check(label, matched, dict(expected=expected, actual=actual, lf_normalized=normalized, crlf_normalized=crlf))
            if matched and actual != expected:
                report['warnings'].append('CRLF/LF-only equivalence accepted: ' + label)
        except OSError as exc:
            check(label, False, str(exc))
    try:
        contract = read_json(args.contract)
        report['contract_sha256'] = sha256(args.contract)
        tracked[str(args.contract.resolve())] = report['contract_sha256']
        check('contract version', contract['format'] == 'hdeg_v3_reference_contract_v1')
        source = Path(__file__).resolve().parent
        for name, digest in contract['source_files'].items():
            verify(child(source, name), digest, 'source:' + name, text_source=True)
        for name, digest in contract['evidence_files'].items():
            verify(child(args.contract.parent, name), digest, 'evidence:' + name)
        verify(args.input_dir / 'manifest.json', contract['input_manifest_sha256'], 'input manifest')
        manifest = read_json(args.input_dir / 'manifest.json')
        check('training input contract', manifest['status'] == 'COMPLETE' and manifest['split'] == 'train'
              and manifest['num_channels'] == 23 and manifest['devices'] == contract['devices']
              and manifest['status_codes'] == contract['status_codes'])
        stop = 0
        for shard in manifest['shards']:
            check('contiguous shard:' + shard['file'], shard['start'] == stop and shard['rows'] == shard['stop'] - shard['start'])
            stop = shard['stop']
            verify(child(args.input_dir, shard['file']), shard['sha256'], 'input shard:' + shard['file'])
        check('row count', stop == manifest['num_rows'] == 1632960)
        for item in manifest['metadata'].values():
            verify(child(args.input_dir, item['file']), item['sha256'], 'input metadata:' + item['file'])
        reports = {}
        for label, root, filename in [('comparison', args.comparison_dir, 'comparison_report.json'), ('residual', args.residual_dir, 'residual_report.json')]:
            verify(root / filename, contract['reports'][label], label + ' report anchor')
            r = read_json(root / filename)
            reports[label] = r
            check(label + ' completed', r['status'] == 'COMPLETE')
            check(label + ' input link', r['manifest_sha256'] == contract['input_manifest_sha256'])
            for name, digest in r['output_hashes'].items():
                verify(child(root, name), digest, label + ':' + name)
            for key, name in contract['run_source_links'].items():
                verify(child(source, name), r['source_hashes'][key], label + ' executed source:' + key, text_source=True)
            old = copy.deepcopy(r['configuration']); now = copy.deepcopy(config)
            old.pop('project_root_dir', None); now.pop('project_root_dir', None)
            ignored = {}
            for key in ('models',):
                a = old.pop(key, None); b = now.pop(key, None)
                if a != b: ignored[key] = dict(reference=a, current=b)
            a = old.get('hdeg', {}).pop('e2e_training', None)
            b = now.get('hdeg', {}).pop('e2e_training', None)
            if a != b: ignored['hdeg.e2e_training'] = dict(reference=a, current=b)
            report.setdefault('unused_configuration_differences', {})[label] = ignored
            check(label + ' diagnostic configuration', old == now)
        split = read_json(args.comparison_dir / 'split_manifest.json')
        verify(args.comparison_dir / 'split_manifest.json', contract['split_sha256'], 'frozen split')
        rows = split['assignments']; ids = [r['index'] for r in rows]
        check('256 unique pairs and 128/128 roles', len(ids) == len(set(ids)) == 256 and Counter(r['role'] for r in rows) == {'update':128, 'probe':128})
        batches = split['batches']; update = [i for b in batches for i in b]
        check('fixed update schedule', len(batches) == 16 and all(len(b)==8 for b in batches)
              and len(set(update)) == 128 and set(update) == {r['index'] for r in rows if r['role']=='update'}
              and split['cycles'] == 8 and split['updates_per_arm'] == 128)
        check('disjoint 31-row spans', all(b-a >= 31 for a,b in zip(sorted(ids), sorted(ids)[1:])))
        verify(args.comparison_dir / 'initial_model.pt', reports['residual']['loaded_checkpoint_sha256'], 'residual original checkpoint link')
        check('original tensor identity recorded', reports['comparison']['initial_state_sha256'] == reports['residual']['original_state_sha256'] == contract['initial_tensor_sha256'])
        check('residual split link', reports['residual']['split_manifest_sha256'] == contract['split_sha256'])
        for name, distribution in [('torch','torch'), ('torch_geometric','torch-geometric'), ('numpy','numpy'), ('yaml','PyYAML'), ('joblib','joblib')]:
            try: report['runtime'][name] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError: report['runtime'][name] = None
        reference = reports['comparison']['framework_versions']
        report['reference_runtime'] = reference
        report['runtime_matches_reference'] = all(report['runtime'].get(k) == v for k,v in reference.items())
        if not report['runtime_matches_reference']:
            report['warnings'].append('Runtime differs or dependencies are missing. Numerical replay is required before any future optimization; this preflight does not perform it.')
        report['missing_preparation_entrypoints'] = [n for n in contract['preparation_entrypoints'] if not (source/n).is_file()]
        report['raw_to_experiment_reproducibility'] = 'NOT_ESTABLISHED'
        report['warnings'].extend(['Existing probes are inspected training-period development data, not independent validation.',
            'Input hashes verify prepared bytes, not upstream freshness or preprocessing causality.',
            'Historical scalar-input Level 1 results do not validate this six-feature representation state.',
            'Legacy trainer is outside the supported contract; its missing import and scalar loader are not repaired here.'])
        for name,digest in list(tracked.items()):
            verify(Path(name), digest, 'unchanged:' + name)
        report['status'] = 'PASS_WITH_LIMITATIONS' if all(c['passed'] for c in report['checks']) else 'BLOCKED'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    with (out/'preflight_report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('V3 experiment-contract preflight:', report['status'])
    print('Report:', out/'preflight_report.json')
    for c in report['checks']:
        if not c['passed']: print('FAILED:', c['name'])
    if 'error' in report: print(report['error'])
    return 0 if report['status']=='PASS_WITH_LIMITATIONS' else 2

if __name__ == '__main__':
    sys.exit(main())
