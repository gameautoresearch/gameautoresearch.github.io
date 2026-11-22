"""Bounded own-run diagnostics and explicit evaluation validity.

No arbitrary path is accepted. Diagnostic source paths are derived from a recorded
operation of this session, never from candidate-controlled strings or private grading data.
"""
import json
import math
from pathlib import Path
import re

LABEL = re.compile(r'[A-Za-z0-9_-]{1,80}\Z')
OPERATION = re.compile(r'[a-f0-9]{32}\Z')
HASH_FIELDS = ('source_sha256', 'config_sha256', 'checkpoint_sha256', 'bundle_sha256')
METRICS = re.compile(r'(?:iteration|iterations|optimizer_updates|trained_samples|samples|natural|natural_games|official_draws|truncations|faults|parameter_change_l2|seconds|elapsed_seconds|loop_seconds|learning_rate|entropy(?:_coef)?|.*_loss|.*_kl|clip_frac|.*grad.*|.*reward.*|.*throughput.*|.*samples_per_second|adv_std|value_target_std)\Z')
MAX_TAIL_BYTES = 512 * 1024


def _own_regular(path, root):
    path, root = Path(path), Path(root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Diagnostics path outside own retained output')
    current = path
    while current != root.parent:
        if current.is_symlink():
            raise ValueError('Linked diagnostics forbidden')
        current = current.parent
    if not path.is_file():
        raise ValueError('Diagnostic regular file required')
    return path


def _tail(path, root, limit=MAX_TAIL_BYTES):
    path = _own_regular(path, root)
    with path.open('rb') as stream:
        size = path.stat().st_size
        stream.seek(max(0, size-limit))
        raw = stream.read(limit)
    # A partial first JSONL line is discarded by the parser below.
    return raw.decode('utf-8', errors='replace')


def metrics(row):
    """Only measured scalar diagnostics; never config/seeds or native state."""
    result, nonfinite = {}, []
    for key, value in row.items() if isinstance(row, dict) else ():
        if not isinstance(key, str) or not METRICS.fullmatch(key):
            continue
        if any(word in key.lower() for word in ('seed','token','path','state')):
            continue
        if type(value) in (int, float):
            if math.isfinite(value):
                result[key] = value
            else:
                nonfinite.append(key)
    return result, nonfinite


def live_output(text, tail=80, samples=12, cut=False):
    """Split a running trainer's printed output into metric rows and other text.

    A printed JSON object is read as a progress row and reduced to the same
    measured scalars a finished run reports; everything else is returned as text.
    `cut` says the text is the end of a longer log.
    """
    rows, other = [], []
    for line in text.splitlines()[1 if cut else 0:]:  # the first line of a cut tail is partial
        row = None
        if line.startswith('{'):
            try:
                row = metrics(json.loads(line))[0]
            except (ValueError, TypeError, RecursionError):
                row = None
        if row:
            rows.append(row)
        elif line.strip():
            other.append(line[:400])
    if len(rows) > samples:
        rows = [rows[round(i * (len(rows) - 1) / (samples - 1))] for i in range(samples)]
    return rows, '\n'.join(other[-tail:])[-4000:]


def evaluation_feedback(summary, execution_success, candidate=None):
    """Transport/worker success is deliberately independent of scored validity."""
    summary = summary if isinstance(summary, dict) else {}
    scheduled = summary.get('scheduled_games')
    natural = summary.get('natural_games', 0)
    faults = summary.get('faults', 0)
    truncations = max(summary.get('administrative_caps',0),summary.get('truncated_games',0),summary.get('truncations',0))
    unstarted = summary.get('unstarted_games', 0)
    complete = (type(scheduled) is int and scheduled > 0 and natural == scheduled
                and faults == 0 and truncations == 0 and unstarted == 0)
    value = summary.get('raw_score_0_100')
    valid = bool(execution_success and complete and type(value) in (int, float)
                 and math.isfinite(value) and 0 <= value <= 100)
    identity = summary.get('candidate_identity')
    expected = dict(candidate or {})
    binding_valid = not expected or bool(expected.get('bundle_sha256')and isinstance(identity,dict)
                        and identity.get('bundle_sha256')==expected['bundle_sha256'])
    if not binding_valid:valid=False
    return dict(execution_success=bool(execution_success), panel_complete=complete,
                score_valid=valid, score_0_100=value if valid else None,
                evaluation_status='complete_valid' if valid else 'incomplete_or_invalid',
                scheduled_games=scheduled, natural_games=natural, faults=faults,
                truncations=truncations, unstarted_games=unstarted,
                candidate_identity=identity, expected_candidate=expected,
                candidate_binding_valid=binding_valid,
                fault_categories=summary.get('fault_categories', {}),
                note='Only a complete fault-free panel bound to this candidate has a valid score.')


def stage_diagnostics(root, stage, record, limit=24):
    """`record` is the session's own stage record; `root` is the session directory."""
    if not isinstance(stage, str) or not LABEL.fullmatch(stage):
        raise ValueError('Simple own stage label required')
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('Diagnostic sample limit must be1..64')
    operation = record.get('operation', '')
    if not isinstance(operation, str) or not OPERATION.fullmatch(operation):
        raise ValueError('Own recorded operation required')
    output = Path(root)/'jobs'/operation/'output'
    # A retained path is a consistency check, never authority to read elsewhere.
    if record.get('output') and Path(record['output']).resolve() != output.resolve():
        raise ValueError('Recorded output differs from own operation')
    result = dict(stage=stage, operation=operation, stage_status=record.get('status'),
                  training_successful=record.get('successful', False),
                  **{key:record[key] for key in HASH_FIELDS if key in record})
    progress = output/'stage/progress.jsonl'
    rows, nonfinite, malformed = [], set(), 0
    if progress.exists():
        _own_regular(progress,output)
        if progress.stat().st_size>MAX_TAIL_BYTES:
            with progress.open('rb')as stream:head=stream.read(MAX_TAIL_BYTES).decode('utf-8',errors='replace')
            # Drop a partial final head line; tail parser drops its partial first.
            raw='\n'.join(head.splitlines()[:-1])+'\n'+_tail(progress,output)
            scope='bounded head and tail; middle rows omitted'
        else:raw=_tail(progress,output);scope='complete retained progress file within byte bound'
        for line in raw.splitlines():
            try:
                row, bad = metrics(json.loads(line))
            except (ValueError, TypeError):
                malformed += 1
                continue
            if row: rows.append(row)
            nonfinite.update(bad)
        if len(rows) > limit:
            indices = sorted({round(i*(len(rows)-1)/(limit-1)) for i in range(limit)}) if limit>1 else [len(rows)-1]
            rows = [rows[i] for i in indices]
    result.update(progress_samples=rows, nonfinite_metric_fields=sorted(nonfinite),
                  malformed_tail_lines=malformed,
                  progress_scope=locals().get('scope','progress not retained; empty samples are not evidence of no learning'))
    final = output/'stage/result.json'
    if final.exists():
        raw = _tail(final, output, 64*1024)
        try:
            values, bad = metrics(json.loads(raw))
            result['training_totals'] = values
            result['nonfinite_metric_fields'] = sorted(nonfinite|set(bad))
        except (ValueError, TypeError):
            result['training_totals_unreadable'] = True
    if rows:
        seconds = sum(row.get('seconds', 0) for row in rows)
        samples = sum(row.get('samples', 0) for row in rows)
        if seconds > 0 and samples:
            result['sampled_decisions_per_training_second'] = samples/seconds
    logs = {}
    for name in ('worker-error.log', 'stage/trainer.log', 'stage/env-service.log'):
        path = output/name
        if path.exists(): logs[name] = _tail(path, output, 12000)
    result['own_log_tails'] = logs
    return result
