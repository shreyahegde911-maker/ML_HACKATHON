"""Freeze, inspect and evaluate reserved trips. All evidence stays in ignored local paths."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathsnap.provenance import freeze, assert_frozen, sha256
from pathsnap.runtime import trip_ids
from pathsnap.data.readers import load_trip_directory, _read
import numpy as np
import httpx

PRIVATE = ROOT / 'results/evaluator'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def private_path(path):
    path = Path(path).resolve()
    allowed = [PRIVATE.resolve(), (ROOT / 'output/private').resolve()]
    if not any(path.is_relative_to(parent) for parent in allowed):
        raise ValueError('Evidence must stay in ignored results/evaluator or output/private')
    return path


def raw_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def inspect_data(args, manifest):
    assert_frozen(manifest)
    supplied = Path(args.root).expanduser().resolve()
    if not supplied.is_dir():
        raise ValueError('Supplied reserved folder does not exist')
    before = raw_hashes(supplied)
    tracks = sorted(supplied.rglob('*.track'))
    parents = {p.parent.parent for p in tracks if p.parent.name == p.stem}
    if len(parents) != 1:
        raise ValueError(f'Expected one root of per-trip folders; found {len(parents)} roots. No data was changed.')
    root = parents.pop()
    ids = trip_ids(root)
    if not ids:
        raise ValueError('No numeric trip folders found')
    rows = []
    for tid in ids:
        directory = root / tid
        row = {'trip_id': tid, 'files': sorted(p.name for p in directory.iterdir() if p.is_file())}
        try:
            trip = load_trip_directory(directory)
            row.update(input_status='valid', **trip['validation'])
            reference = directory / f'{tid}.route'
            if not reference.exists():
                row['reference_status'] = 'missing'
            else:
                try:
                    values = _read(reference, 1, integer=True).ravel()
                    edges = trip['arcs'][['from_node_id', 'to_node_id']].to_numpy(dtype=int)
                    if ((values < 0) | (values >= len(edges))).any():
                        raise ValueError('Reference arc ID out of bounds')
                    endpoints = edges[values]
                    row.update(reference_status='valid', reference_occurrences=len(values),
                               reference_discontinuities=int((endpoints[:-1, 1] != endpoints[1:, 0]).sum()))
                except (OSError, ValueError) as error:
                    row.update(reference_status='invalid', reference_error=str(error))
        except (OSError, ValueError) as error:
            row.update(input_status='invalid', input_error=str(error))
        rows.append(row)
    after = raw_hashes(supplied)
    if before != after:
        raise RuntimeError('Raw data changed during inspection')
    non_trip_files = [str(p.relative_to(supplied)) for p in supplied.rglob('*') if p.is_file() and p.suffix not in {'.track', '.nodes', '.arcs', '.route'}]
    result = {'inspected_at_utc': datetime.now(timezone.utc).isoformat(), 'supplied_folder': str(supplied),
              'trip_root': str(root), 'trip_count': len(ids), 'trip_ids': ids, 'trips': rows,
              'other_files': non_trip_files, 'raw_sha256': before,
              'originals_unchanged': True, 'freeze_pipeline_sha256': manifest['pipeline_sha256']}
    save(PRIVATE / 'inspection.json', result)
    print(json.dumps({k: result[k] for k in ['trip_root', 'trip_count', 'trip_ids', 'other_files', 'originals_unchanged']}, indent=2))
    print('Input validation:', Counter(r['input_status'] for r in rows))
    print('Reference validation:', Counter(r.get('reference_status', 'not validated') for r in rows))


def no_reference_leaks(value):
    banned = {'reference_arc_id', 'alignment_status', 'alignment_distance_m', 'reference_route', 'reference_geometry', 'ground_truth', 'reference_section_id', 'label_reliable', 'route_position'}
    if isinstance(value, dict):
        if banned.intersection(value):
            raise AssertionError('Reference fields exposed in API response')
        for child in value.values():
            no_reference_leaks(child)
    elif isinstance(value, list):
        for child in value:
            no_reference_leaks(child)


def aggregate(rows):
    valid = [r for r in rows if r.get('checks_passed')]
    output = {'trip_count': len(rows), 'successful_trips': len(valid),
              'failed_trips': len(rows) - len(valid),
              'total_observations': sum(r['observations'] for r in valid),
              'http_runtime_s': sum(r['http_runtime_s'] for r in rows),
              'trips_with_metrics': sum(r.get('metrics') is not None for r in valid),
              'label_basis': 'Agreement with provisional derived labels from the frozen ordered-reference alignment; not an official organiser score'}
    for name in ['classifier', 'hmm']:
        items = [r['metrics'][name] for r in valid if r.get('metrics')]
        eligible = sum(m.get('eligible_observations', 0) for m in items)
        correct = sum(round(m['top1_accuracy'] * m['eligible_observations']) for m in items if m.get('top1_accuracy') is not None)
        pairs = sum(m['continuity_pairs'] for m in items if m.get('direct_step_continuity') is not None)
        continuous = sum(round(m['direct_step_continuity'] * m['continuity_pairs']) for m in items if m.get('direct_step_continuity') is not None)
        set_f1 = [m['arc_set_f1'] for m in items if m.get('arc_set_f1') is not None]
        output[name] = {'eligible_observations': eligible, 'correct_observations': correct,
                        'top1_accuracy': correct / eligible if eligible else None,
                        'alignment_coverage': eligible / output['total_observations'] if output['total_observations'] else None,
                        'continuity_pairs': pairs, 'direct_step_continuity': continuous / pairs if pairs else None,
                        'mean_trip_arc_set_f1': sum(set_f1) / len(set_f1) if set_f1 else None}
    return output


def evaluate(args, manifest):
    assert_frozen(manifest)
    inspection = json.loads((PRIVATE / 'inspection.json').read_text())
    root = Path(inspection['trip_root'])
    if raw_hashes(Path(inspection['supplied_folder'])) != inspection['raw_sha256']:
        raise RuntimeError('Reserved input hashes changed since inspection')
    records = []
    with httpx.Client(base_url=args.url, timeout=600) as client:
        health = client.get('/health').raise_for_status().json()
        catalog = client.get('/trips').raise_for_status().json()
        system = client.get('/system').raise_for_status().json()
        if health['mode'] != 'reserved-evaluator' or catalog['trips'] != inspection['trip_ids']:
            raise RuntimeError('API is not serving exactly the inspected reserved catalog')
        if system['pipeline_sha256'] != manifest['pipeline_sha256']:
            raise RuntimeError('Running API does not match the frozen pipeline')
        for tid in catalog['trips']:
            row = {'trip_id': tid}
            tick = time.perf_counter()
            try:
                # The loader reads observations/network only. The API performs reference evaluation after prediction.
                trip = load_trip_directory(root / tid)
                response = client.post('/predict', json={'trip_id': tid})
                row['http_status'] = response.status_code
                response.raise_for_status()
                data = response.json()
                no_reference_leaks(data)
                assert data['trip_id'] == tid and data['mode'] == 'reserved-evaluator'
                predictions = data['predictions']
                assert len(predictions) == len(trip['observations']) == data['observation_count']
                assert [p['observation_id'] for p in predictions] == trip['observations'].observation_id.tolist()
                n_arcs = len(trip['arcs'])
                assert all(p['arc_id'] is None or isinstance(p['arc_id'], int) and 0 <= p['arc_id'] < n_arcs for p in predictions)
                edges = trip['arcs'][['from_node_id', 'to_node_id']].to_numpy(dtype=int)
                for run in data['route_runs']:
                    assert run and all(isinstance(arc, int) and 0 <= arc < n_arcs for arc in run)
                    assert all(edges[a, 1] == edges[b, 0] for a, b in zip(run, run[1:]))
                assert len(data['decoded_segments']) == len({arc for run in data['route_runs'] for arc in run})
                assert len(data['map_geojson']['gps']['coordinates']) == len(predictions)
                assert data['decoder_settings'] == manifest['configuration']['decoder']
                breaks = {event['observation_id']: event['reason'] for event in data['breaks']}
                times = trip['observations'].timestamp.to_numpy()
                gap_limit = manifest['configuration']['decoder']['gap_s']
                gap_ids = (np.flatnonzero((np.diff(times) <= 0) | (np.diff(times) > gap_limit)) + 1).tolist()
                for index in gap_ids:
                    if predictions[index - 1]['arc_id'] is not None and predictions[index]['arc_id'] is not None:
                        assert breaks.get(index) == 'timestamp_gap'
                if data['metrics'] is None:
                    assert data['label_basis'] == 'reference unavailable' and data.get('evaluation_warning')
                row.update(checks_passed=True, observations=len(predictions), null_predictions=sum(p['arc_id'] is None for p in predictions),
                           route_runs=len(data['route_runs']), gap_reset_events=len(data['breaks']), timestamp_gaps=len(gap_ids),
                           server_processing_s=data['inference_s'], metrics=data['metrics'], evaluation_warning=data.get('evaluation_warning'),
                           arc_id_bounds='passed', route_connectivity='passed', gap_handling='passed', reference_leak_check='passed')
                # Prediction-only artifact; still private because the reserved GPS/network is private.
                save(PRIVATE / 'predictions' / f'{tid}.json', {'trip_id': tid, 'predictions': predictions, 'route_runs': data['route_runs'], 'breaks': data['breaks']})
            except Exception as error:
                row.update(checks_passed=False, error=f'{type(error).__name__}: {error}')
            row['http_runtime_s'] = time.perf_counter() - tick
            records.append(row)
            save(PRIVATE / 'api_results.partial.json', records)
            print(json.dumps({k: row.get(k) for k in ['trip_id', 'http_status', 'checks_passed', 'observations', 'http_runtime_s', 'error']}), flush=True)
    unchanged = raw_hashes(Path(inspection['supplied_folder'])) == inspection['raw_sha256']
    assert_frozen(manifest)
    result = {'evaluated_at_utc': datetime.now(timezone.utc).isoformat(), 'freeze': manifest,
              'official_scoring_procedure': args.scoring_note, 'aggregate': aggregate(records), 'trips': records,
              'originals_unchanged': unchanged, 'pipeline_unchanged': True,
              'health': health, 'catalog': catalog, 'server_system': system,
              'privacy': 'Local evidence only; no reserved data or label-derived artifacts may be published without explicit permission'}
    save(PRIVATE / 'final_evaluation.json', result)
    print(json.dumps(result['aggregate'], indent=2))
    if not unchanged or any(not row['checks_passed'] for row in records):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'inspect', 'run'])
    parser.add_argument('--root', default=str(ROOT / 'evaluator_test'))
    parser.add_argument('--manifest', default=str(PRIVATE / 'freeze.json'))
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--scoring-note', default='No official scoring procedure supplied; frozen provisional alignment and sequence_metrics used.')
    args = parser.parse_args()
    manifest_path = private_path(args.manifest)
    if args.action == 'freeze':
        if manifest_path.exists():
            raise RuntimeError('Freeze already exists; never silently overwrite a prior evaluation freeze')
        if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', 'src', 'scripts', 'ui', 'configs/decoder.json', 'requirements.txt'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Commit outstanding implementation fixes before freezing')
        result = freeze()
        save(manifest_path, result)
        print(json.dumps({k: result[k] for k in ['created_at_utc', 'git_commit', 'model_sha256', 'pipeline_sha256', 'configuration_sha256']}, indent=2))
        return
    manifest = json.loads(manifest_path.read_text())
    if args.action == 'inspect':
        inspect_data(args, manifest)
    else:
        evaluate(args, manifest)


if __name__ == '__main__':
    main()
