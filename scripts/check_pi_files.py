#!/usr/bin/env python3
"""Read-only inventory. A candidate classification never authorizes deletion."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import urllib.request


@dataclass
class AuditContext:
    repo_root: Path
    model_root: Path
    data_root: Path
    active_models: set[Path] = field(default_factory=set)
    references: dict[str, list[str]] = field(default_factory=dict)
    protected_paths: set[Path] = field(default_factory=set)
    model_evidence: str = 'configuration (conservative)'


def beneath(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve()) or path.absolute().is_relative_to(root.absolute())


def classify(path: Path, ctx: AuditContext) -> tuple[str, str]:
    resolved = path.resolve()
    name = path.name.lower()
    if (beneath(path, ctx.data_root) or resolved in ctx.protected_paths
            or (name.startswith('.env') and name != '.env.example')
            or any(word in name for word in ('service-account', 'serviceaccountkey', 'firebase-adminsdk'))
            or any(part.lower() in ('secrets', 'enrollments', 'unknown', 'calibration') for part in path.parts)
            or re.search(r'\.(sqlite3?|db)(?:[.-]|$)', name)):
        return 'KEEP_PRIVATE_DATA', 'Configuration, credentials or biometric data; preserve including backups'
    if resolved in {p.resolve() for p in ctx.active_models}:
        return 'KEEP_ACTIVE_MODEL', f'Selected model or alias; evidence: {ctx.model_evidence}'
    if name.endswith('.onnx') or '.onnx.' in name:
        return 'REVIEW_MODEL', 'Potential reference/rollback model; benchmark and confirm selection before archiving'
    if '.git' in path.parts or '.venv' in path.parts:
        return 'KEEP_DEPENDENCY', 'Git history or installed runtime dependencies'
    if any(p in ('__pycache__', '.pytest_cache', 'build', 'dist') or p.endswith('.egg-info') for p in path.parts) or name.endswith('.pyc'):
        return 'REGENERABLE', 'Generated cache/build metadata; inspect runtime and package use before removal'
    references = ctx.references.get(path.name, []) + ctx.references.get(str(path), [])
    if references:
        return 'KEEP_REFERENCED', 'Referenced by: ' + ', '.join(sorted(set(references))[:5])
    if re.search(r'\.bak(?:[._-]|$)|\.backup(?:[._-]|$)|~$', name):
        return 'ARCHIVE_CANDIDATE', 'No static project reference found; preserve a backup outside the runtime directory'
    if name.endswith('.deb'):
        return 'REVIEW_PACKAGE', 'Installer package; confirm package installation and recovery needs before archiving'
    if beneath(path, ctx.repo_root):
        rel = path.absolute().relative_to(ctx.repo_root.absolute())
        if rel.parts and rel.parts[0] in ('src', 'scripts', 'deploy') or name in ('pyproject.toml', '.env.example', '.gitignore', 'readme.md'):
            return 'KEEP_RUNTIME', 'Current source, runtime configuration template or deployment tool'
        if rel.parts and rel.parts[0] in ('tests', 'docs'):
            return 'KEEP_VALIDATION', 'Regression tests or operational documentation'
    if str(path).startswith(('/etc/', '/usr/', '/bin/', '/lib/')):
        return 'KEEP_SYSTEM', 'System/service file; check package owner and unit references'
    return 'KEEP_REVIEW', 'Usage not established; inspect without executing or deleting the file'


def safe_config(path: Path) -> dict[str, str]:
    """Read only path/selectors; never source the shell file or print secret values."""
    selected = {}
    allowed = {'MODEL_ROOT', 'MODEL_PACK', 'RECOGNIZER_MODEL', 'DATA_ROOT', 'FIREBASE_SERVICE_ACCOUNT'}
    if path.is_file():
        for line in path.read_text(errors='replace').splitlines():
            match = re.match(r'^\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.*?)\s*$', line)
            if match and match[1] in allowed:
                value = match[2]
                if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                    value = value[1:-1]
                else:
                    value = value.split(' #', 1)[0].strip()
                selected[match[1]] = value
    for key in allowed:
        if key in os.environ:
            selected[key] = os.environ[key]
    return selected


def references(repo: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    candidates = [p.name for p in walk(repo, False) if p.is_file()]
    suspect = {name for name in candidates if '.bak' in name or '.backup' in name or name.endswith('.deb')}
    for root in (repo / 'src', repo / 'scripts', repo / 'deploy', repo / 'docs'):
        for path in walk(root, False):
            if not path.is_file() or path.suffix not in {'.py', '.sh', '.md', '.service', '.toml'} or path.stat().st_size > 1024 * 1024:
                continue
            content = path.read_text(errors='replace')
            for name in suspect:
                if name != path.name and name in content:
                    found.setdefault(name, []).append(str(path.relative_to(repo)))
    return found


def walk(root: Path, include_caches: bool):
    if not root.exists() and not root.is_symlink():
        return
    if not root.is_dir() or root.is_symlink():
        yield root
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(current) / name
            skip = name in ('.git', '.venv') or (not include_caches and (name in ('__pycache__', '.pytest_cache', 'build', 'dist') or name.endswith('.egg-info')))
            if path.is_symlink() or skip:
                yield path
                dirs.remove(name)
        for name in files:
            yield Path(current) / name


def metadata(path: Path, ctx: AuditContext, hash_files: bool) -> dict:
    decision, reason = classify(path, ctx)
    row = {'path': str(path), 'decision': decision, 'reason': reason}
    try:
        info = path.lstat()
        row.update(bytes=info.st_size, modified_at=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                   mode=stat.filemode(info.st_mode), symlink_target=str(path.resolve()) if path.is_symlink() else None)
        if hash_files and path.is_file() and not decision.startswith('KEEP_PRIVATE'):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            row['sha256'] = digest.hexdigest()
    except OSError as exc:
        row['error'] = str(exc)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--health-url', help='Optional local /health URL to establish actual selected models')
    parser.add_argument('--file', type=Path, help='Inspect one file; no file contents are printed')
    parser.add_argument('--extra-root', type=Path, action='append', default=[])
    parser.add_argument('--include-caches', action='store_true')
    parser.add_argument('--hash', action='store_true', help='SHA256 non-private files (may read large ONNX files)')
    parser.add_argument('--output', type=Path, help='Write JSON report with mode 0600')
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    config = safe_config(repo / '.env')
    def rooted(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else repo / path).resolve()
    ctx = AuditContext(repo, rooted(args.model_root or config.get('MODEL_ROOT', '.')),
                       rooted(args.data_root or config.get('DATA_ROOT', './data')))
    pack = ctx.model_root / 'models' / config.get('MODEL_PACK', 'pi4_fp32')
    ctx.active_models.update(p.resolve() for p in pack.glob('*.onnx'))
    selection = config.get('RECOGNIZER_MODEL', 'auto')
    if selection == 'auto':
        ctx.active_models.add((ctx.model_root / 'models' / 'w600k_mbf_qdq_int8.onnx').resolve())
    elif selection != 'pack':
        path = Path(selection).expanduser()
        ctx.active_models.add((path if path.is_absolute() else ctx.model_root / path).resolve())
    if config.get('FIREBASE_SERVICE_ACCOUNT'):
        ctx.protected_paths.add(rooted(config['FIREBASE_SERVICE_ACCOUNT']))
    warnings = []
    if args.health_url:
        parsed = urllib.parse.urlparse(args.health_url)
        if parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            parser.error('--health-url must point to localhost')
        try:
            with urllib.request.urlopen(args.health_url, timeout=3) as response:
                health = json.load(response)
            files = health.get('model_info', {}).get('files', [])
            if not files:
                warnings.append('Runtime has no model_info.files; using conservative configuration evidence')
            else:
                ctx.active_models = {Path(p).resolve() for p in files}
                ctx.model_evidence = 'local runtime /health'
        except (OSError, ValueError) as exc:
            warnings.append(f'Health unavailable; using conservative configuration evidence: {exc}')
    ctx.references = references(repo)
    if args.file:
        path = args.file.expanduser()
        paths = [(path if path.is_absolute() else repo / path).absolute()]
    else:
        roots = [repo, ctx.model_root / 'models', ctx.data_root, *args.extra_root]
        paths = sorted({p.absolute() for root in roots for p in walk(root.expanduser(), args.include_caches)}, key=str)
    rows = [metadata(path, ctx, args.hash) for path in paths]
    if args.file:
        try:
            owner = subprocess.run(['dpkg-query', '-S', str(paths[0])], capture_output=True, text=True, timeout=5)
            if owner.returncode == 0:
                rows[0]['package_owner'] = owner.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'read_only': True,
               'model_evidence': ctx.model_evidence, 'warnings': warnings, 'files': rows}
    if args.output:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.chmod(args.output, 0o600)
    if args.file:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print('decision\tbytes\tpath\treason')
        for row in rows:
            print('\t'.join(str(row.get(k, '')).replace('\t', ' ').replace('\n', ' ') for k in ('decision', 'bytes', 'path', 'reason')))
        for warning in warnings:
            print('WARNING\t' + warning)
    return 1 if any('error' in row for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
