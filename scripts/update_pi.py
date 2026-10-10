#!/usr/bin/env python3
"""Stage and verify a Git commit before updating the existing Pi checkout.
Default: check only. --apply backs up local work and activates after successful checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import sqlite3
import subprocess
import tarfile
import time
import urllib.request


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def require_compatible(report: dict, allow_incompatible: bool) -> None:
    count = report['catalog']['incompatible_users']
    if count and not allow_incompatible:
        raise RuntimeError(f'{count} active users have incompatible embedding identities. '
                           'Current service is preserved. Validate migration or plan re-enrollment; '
                           '--allow-incompatible explicitly accepts that these users will not match.')


def preserve_collisions(repo: Path, commit: str, backup: Path, moved: list[str] | None = None) -> list[str]:
    incoming = set(git(repo, 'ls-tree', '-rz', '--name-only', commit).split('\0'))
    untracked = git(repo, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')
    if moved is None: moved = []
    for name in sorted(set(untracked) & incoming):
        if not name: continue
        target = backup / 'collisions' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(repo / name), str(target))
        moved.append(name)
        (backup / 'moved-untracked.json').write_text(json.dumps(moved, indent=2))
    return moved


RUNTIME_KEYS = {'DATA_ROOT', 'MODEL_ROOT', 'MODEL_PACK', 'RECOGNIZER_MODEL', 'FACE_DET_SIZE', 'API_ENABLED', 'API_PORT'}


def service_environment(repo: Path) -> dict[str, str]:
    def show(property_name):
        return subprocess.check_output(['systemctl', 'show', 'sic-pi4-attendance',
                                        '--property=' + property_name, '--value'], text=True).strip()
    if Path(show('WorkingDirectory')).resolve() != repo:
        raise RuntimeError('Service WorkingDirectory differs from --repo; source activation is not safe')
    files = show('EnvironmentFiles')
    if files != str(repo / '.env') + ' (ignore_errors=no)':
        raise RuntimeError('Service must use exactly the current repo .env; inspect EnvironmentFiles before updating')
    # Values may include secrets; keep them in memory and never print this property.
    return dict(item.split('=', 1) for item in shlex.split(show('Environment')) if '=' in item)


def update_config(path: Path, changes: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    for key, value in changes.items():
        found = False
        replacement = []
        for line in lines:
            if re.match(rf'^\s*(?:export\s+)?{re.escape(key)}\s*=', line):
                if not found: replacement.append(f'{key}={value}')
                found = True
            else: replacement.append(line)
        if not found: replacement.append(f'{key}={value}')
        lines = replacement
    path.write_text('\n'.join(lines) + '\n')
    path.chmod(0o600)


def backup_work(repo: Path, backup: Path) -> None:
    names = set(git(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0'))
    if (repo / '.env').exists(): names.add('.env')
    with tarfile.open(backup / 'local-files.tar.gz', 'w:gz') as archive:
        for name in sorted(names):
            if name and ((repo / name).exists() or (repo / name).is_symlink()):
                archive.add(repo / name, arcname=name, recursive=False)
    patch = subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary', 'HEAD'])
    (backup / 'local-changes.patch').write_bytes(patch)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/home/sic4b/Smart-Attendance-Pi4-multiple-cams'))
    parser.add_argument('--ref', default='origin/fix/pi-runtime-and-file-audit')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--allow-incompatible', action='store_true')
    parser.add_argument('--preview-transport', choices=('http', 'https'), help='Required with --apply; selects cookie Secure flag')
    args = parser.parse_args(argv)
    if args.apply and args.preview_transport is None: parser.error('--apply requires --preview-transport http or https')
    os.umask(0o077)
    repo = args.repo.expanduser().resolve()
    python = repo / '.venv' / 'bin' / 'python'
    if not python.exists(): parser.error('Existing Pi .venv/bin/python is required')
    if not (repo / '.env').is_file(): parser.error('Existing private .env is required; no template is installed over it')
    commit = git(repo, 'rev-parse', '--verify', args.ref + '^{commit}')
    old_commit = git(repo, 'rev-parse', 'HEAD')
    old_branch = git(repo, 'branch', '--show-current')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = repo.parent / 'sic-backups' / stamp
    backup.mkdir(parents=True, mode=0o700)
    stage = backup / 'candidate'; stage.mkdir(mode=0o700)
    subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', '--output', str(backup / 'candidate.tar'), commit], check=True)
    with tarfile.open(backup / 'candidate.tar') as archive:
        for item in archive.getmembers():
            if not (item.isfile() or item.isdir()) or not (stage / item.name).resolve().is_relative_to(stage):
                raise RuntimeError('Unexpected archive entry')
        archive.extractall(stage, filter='data')
    unit_env = service_environment(repo)
    env = {**{k: v for k, v in os.environ.items() if k not in RUNTIME_KEYS},
           **unit_env, 'PYTHONPATH': str(stage / 'src')}
    print(f'Candidate: {commit}\nCheck reports and backup directory: {backup}', flush=True)
    # Validate against the candidate constraints, not stale editable metadata
    # (the old project required firebase-admin<7 on a Pi already running 7.x).
    dependency_check = '''
import sys, tomllib
from pathlib import Path
from importlib.metadata import version
from pip._vendor.packaging.requirements import Requirement
project = tomllib.loads(Path(sys.argv[1]).read_text())["project"]
for item in project["dependencies"] + project.get("optional-dependencies", {}).get("test", []):
    requirement = Requirement(item)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    installed = version(requirement.name)
    if installed not in requirement.specifier:
        raise SystemExit(f"Install dependency before activation: {item}; installed {installed}")
'''
    subprocess.run([str(python), '-c', dependency_check, str(stage / 'pyproject.toml')], cwd=stage, check=True)
    subprocess.run([str(python), '-c', 'import aiortc, av, onnx, onnxruntime, firebase_admin'], env=env, cwd=stage, check=True)
    subprocess.run([str(python), '-c', 'from importlib.metadata import version; '
                    'assert int(version("setuptools").split(".")[0]) >= 68; version("wheel")'], cwd=stage, check=True)
    subprocess.run([str(python), '-m', 'pytest', '-q', '--capture=sys'], env=env, cwd=stage, check=True)
    report_path = backup / 'model-preflight.json'
    subprocess.run([str(python), '-m', 'pi4_attendance.tools.model_info', '--env', str(repo / '.env'),
                    '--output', str(report_path)], env=env, cwd=repo, check=True)
    report = json.loads(report_path.read_text())
    require_compatible(report, args.allow_incompatible)
    if not args.apply:
        print('Checks passed. Current checkout and service were not changed. Use --apply --preview-transport http|https to activate.')
        return 0
    # All model/catalog/tests gates finish before stopping the service or moving local files.
    backup_work(repo, backup)
    shutil.copy2(repo / '.env', backup / 'original.env')
    database = Path(report['data_root']) / 'smart_attendance.sqlite3'
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'sic-pi4-attendance']).returncode == 0
    subprocess.run(['sudo', '-v'], check=True)
    stash = None; moved = []; switched = False
    deployment_branch = 'pi-deploy/' + stamp
    try:
        subprocess.run(['sudo', 'systemctl', 'stop', 'sic-pi4-attendance'], check=True)
        # Enrollment can arrive during the earlier checks. Freeze and recheck
        # before any source/config mutation; never relabel old embeddings.
        code = ('import json,sys; from pathlib import Path; '
                'from pi4_attendance.tools.model_info import catalog_compatibility; '
                'print(json.dumps(catalog_compatibility(Path(sys.argv[1]),sys.argv[2])))')
        frozen = subprocess.check_output([str(python), '-c', code, str(database), report['model_version']],
                                         env=env, cwd=repo, text=True)
        report['catalog'] = json.loads(frozen)
        report_path.write_text(json.dumps(report, indent=2))
        require_compatible(report, args.allow_incompatible)
        if database.exists():
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(backup / 'smart_attendance.sqlite3') as destination:
                    source.backup(destination)
        if git(repo, 'status', '--porcelain', '--untracked-files=no'):
            subprocess.run(['git', '-C', str(repo), 'stash', 'push', '-m', 'sic-before-' + stamp], check=True)
            stash = git(repo, 'rev-parse', 'stash@{0}')
        preserve_collisions(repo, commit, backup, moved)
        state = {'previous_branch': old_branch, 'previous_commit': old_commit, 'candidate_commit': commit,
                 'stash': stash, 'moved_untracked': moved, 'deployment_branch': deployment_branch}
        (backup / 'deployment.json').write_text(json.dumps(state, indent=2))
        subprocess.run(['git', '-C', str(repo), 'switch', '-c', deployment_branch, commit], check=True)
        switched = True
        changes = {'PREVIEW_COOKIE_SECURE': 'true' if args.preview_transport == 'https' else 'false'}
        if report['model_info'].get('selection') == 'auto':
            changes['RECOGNIZER_MODEL'] = report['model_info']['files'][-1] if report['model_info']['recognizer_source'] == 'external' else 'pack'
        update_config(repo / '.env', changes)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--no-index', '-e', '.'], cwd=repo, check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], cwd=repo, check=True)
        subprocess.run(['sudo', 'systemctl', 'start', 'sic-pi4-attendance'], check=True)
        port = report['api_port']
        if report['api_enabled']:
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('model_version') == report['model_version']: break
                except (OSError, ValueError): pass
                if time.monotonic() >= deadline: raise RuntimeError('New service did not publish expected model identity')
                time.sleep(1)
        subprocess.run(['systemctl', 'is-active', '--quiet', 'sic-pi4-attendance'], check=True)
        print(f'Activated {commit}. Local edits/backups are preserved at {backup}; Git stash is retained. Inspect camera_status in /health.')
        if report['catalog']['incompatible_users']:
            print('Explicit override used: incompatible users remain stored and cannot match until validated migration or re-enrollment.')
    except BaseException:
        print(f'Activation failed; restoring source/configuration. Recovery artifacts: {backup}', flush=True)
        subprocess.run(['sudo', 'systemctl', 'stop', 'sic-pi4-attendance'], check=False)
        if switched:
            restore = ['switch', old_branch] if old_branch else ['switch', '--detach', old_commit]
            subprocess.run(['git', '-C', str(repo), *restore], check=True)
        if stash: subprocess.run(['git', '-C', str(repo), 'stash', 'apply', '--index', stash], check=True)
        for name in moved:
            target = repo / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup / 'collisions' / name), str(target))
        shutil.copy2(backup / 'original.env', repo / '.env')
        restored = subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--no-index', '-e', '.'], cwd=repo, check=False) if switched else None
        if active: subprocess.run(['sudo', 'systemctl', 'start', 'sic-pi4-attendance'], check=True)
        if restored is not None and restored.returncode:
            print('Old service restart attempted; package metadata restoration failed. Inspect the backup and pip output.')
        raise
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
