#!/usr/bin/env python3
"""Audit readable wiki catalogs, pinned-source hashes and standalone offline exports."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

from research_tasks import ROOT, catalog, check_rules, export_rulebook, rulebook_files


def audit():
    errors = check_rules()
    rows = []
    for game, entry in catalog().items():
        base = ROOT / entry['autoresearch'] / 'task/game_rules'
        maintenance = ROOT / entry['autoresearch'] / 'wiki_maintenance/metadata'
        coverage_path = maintenance / 'coverage.json'
        if not coverage_path.is_file():
            continue
        coverage = json.loads(coverage_path.read_text())
        files = rulebook_files(game)
        source_hashes = {}
        for key in ('source_sha256', 'sources_sha256'):
            source_hashes.update(coverage.get(key, {}))
        reference = maintenance / 'reference.json'
        if reference.is_file():
            ref = json.loads(reference.read_text())
            source_hashes.update(ref.get('sha256', {}))
        for name, expected in source_hashes.items():
            path = ROOT / name
            if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
                errors.append(f'{game}: invalid/missing provenance source {name}')
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'{game}: provenance source hash changed: {name}')
        catalogs = []
        for item in coverage.get('catalogs', []):
            expected = item.get('expected', item.get('entries'))
            documented = item.get('documented')
            if isinstance(expected, int) and isinstance(documented, int) and documented != expected:
                errors.append(f'{game}: catalog count mismatch: {item["name"]}')
            catalogs.append({k:v for k,v in item.items() if k not in {'ids'}})
        export_checks = {}
        if not any(error.startswith(game + ':') for error in errors):
            with tempfile.TemporaryDirectory(prefix='wiki-audit-') as temp:
                out = Path(temp) / game
                manifest = export_rulebook(game, out)['files_sha256']
                actual = {p.relative_to(out).as_posix() for p in out.rglob('*')
                          if p.is_file()}
                if actual != set(manifest):
                    errors.append(f'{game}: incomplete export manifest')
                for name, expected in manifest.items():
                    if hashlib.sha256((out/name).read_bytes()).hexdigest() != expected:
                        errors.append(f'{game}: exported file hash mismatch: {name}')
                link_count = 0
                for page in out.rglob('*.md'):
                    text = re.sub(r'```.*?```', '', page.read_text(), flags=re.S)
                    for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', text):
                        target = link.split('#', 1)[0]
                        if not target or '://' in target or target.startswith('mailto:'):
                            continue
                        link_count += 1
                        resolved = (page.parent / target).resolve()
                        if not resolved.is_relative_to(out.resolve()) or not resolved.exists():
                            errors.append(f'{game}: standalone export link failed: {page.relative_to(out)}: {target}')
                export_checks = {'files_hashed': len(manifest), 'local_links_checked': link_count}
        rows.append({'game': game, 'name': entry['name'], 'mode': coverage.get('mode'),
                     'declared_status': coverage.get('status'),
                     'markdown_files': sum(name.endswith('.md') for name in files),
                     'readable_files': len(files), 'readable_bytes': sum((base/name).stat().st_size for name in files),
                     'catalogs': catalogs, 'provenance_hashes_checked': len(source_hashes),
                     'standalone_export': export_checks,
                     'limitations': coverage.get('gaps', coverage.get('known_gaps', coverage.get('limitations', []))),
                     'coverage_file': coverage_path.relative_to(ROOT).as_posix()})
    return {'schema_version': 1, 'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'scope': 'Offline game-mode wiki documentation; no simulation, training, provider calls or private grading.',
            'semantic_limit': 'Catalog, source-identity and packaging checks do not prove every prose assertion or every simulator interaction.',
            'status': 'passed' if not errors and len(rows)==len(catalog()) else 'incomplete', 'games': rows, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.out:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status':result['status'], 'games':[{'game':r['game'], 'markdown_files':r['markdown_files'],
                     'source_hashes':r['provenance_hashes_checked'], **r['standalone_export']} for r in result['games']],
                      'errors':result['errors']}, indent=2))
    raise SystemExit(result['status'] != 'passed')


if __name__ == '__main__':
    main()
