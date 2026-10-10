#!/usr/bin/env python3
"""Update WebRTC and add a read-only log route; leave inference/models/data intact."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patch_event_feed(source: str) -> str:
    tree = ast.parse(source)
    if any(isinstance(node, ast.Constant) and node.value == '/api/recognition-events'
           for node in ast.walk(tree)):
        return source
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'create_app']
    if len(functions) != 1:
        raise ValueError('Cannot identify create_app; leave the API unchanged')
    function = functions[0]
    final = function.body[-1]
    if (not isinstance(final, ast.Return) or not isinstance(final.value, ast.Name)
            or final.value.id != 'app' or not function.args.args or function.args.args[0].arg != 'runtime'
            or not any(isinstance(node, ast.Name) and node.id == 'require_admin'
                       for node in ast.walk(function))):
        raise ValueError('Unknown API factory shape; leave the API unchanged')
    route = '''    @app.get("/api/recognition-events")
    def media_recognition_events(limit: int = 100, _: str = Depends(require_admin)) -> list[dict]:
        import sqlite3
        database = runtime.store.database_path.resolve()
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=2) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                'SELECT event.*, users.student_id FROM recognition_events AS event '
                'LEFT JOIN users ON users.user_id=event.user_id '
                'ORDER BY event.observed_at DESC,event.observation_id DESC LIMIT ?',
                (max(1, min(limit, 500)),)).fetchall()
        return [{'observation_id': row['observation_id'], 'occurred_at': row['observed_at'],
                 'camera_ids': [row['camera_id']], 'status': row['status'],
                 'display_status': 'accepted' if row['status'] == 'accepted' else
                    'unknown' if row['reason'] in {'no_candidate', 'similarity_below_threshold'} else 'rejected',
                 'name': row['name'] if row['status'] == 'accepted' else None,
                 'student_id': row['student_id'] if row['status'] == 'accepted' else None,
                 'similarity': row['similarity_scaled'] / 10000,
                 'reason': row['reason'], 'model_version': row['model_version']} for row in rows]

'''
    lines = source.splitlines(keepends=True)
    lines.insert(final.lineno - 1, route)
    result = ''.join(lines)
    compile(result, 'api.py', 'exec')
    return result


def update(repo: Path, source: Path, *, apply: bool = False):
    repo = repo.resolve(strict=True)
    target = repo / 'src' / 'pi4_attendance' / 'webrtc.py'
    if target.is_symlink() or not target.is_file():
        raise ValueError('Current webrtc.py must be a regular file')
    current = target.read_bytes()
    incoming = source.read_bytes()
    tree = ast.parse(incoming, filename=str(source))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if not {'PiVideoTrack', 'WebRTCManager'}.issubset(classes):
        raise ValueError('Candidate is not the expected WebRTC module')
    for required in (b'WEBRTC_FIREBASE_SERVICE_ACCOUNT', b'_expiry_watchdog', b'sic-webrtc'):
        if required not in incoming:
            raise ValueError('Candidate lacks isolated media credentials or peer expiration')
    result = {'target': str(target), 'previous_sha256': digest(current),
              'candidate_sha256': digest(incoming), 'changed': incoming != current,
              'applied': False, 'restart_required': False}
    plan = [(target, current, incoming)] if incoming != current else []
    api_target = target.with_name('api.py')
    if api_target.exists():
        if api_target.is_symlink():
            raise ValueError('Current api.py must be a regular file')
        old_api = api_target.read_bytes()
        new_api = patch_event_feed(old_api.decode('utf-8')).encode('utf-8')
        result['event_feed_added'] = new_api != old_api
        if new_api != old_api:
            plan.append((api_target, old_api, new_api))
    result['changed'] = bool(plan)
    if not apply or not plan:
        return result
    backup_root = repo.parent / 'sic-media-backups'
    backup_root.mkdir(mode=0o700, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(
        prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=backup_root))
    backups = {}
    for path, before, after in plan:
        backup = backup_dir / path.name
        shutil.copy2(path, backup)
        if backup.read_bytes() != before:
            raise ValueError('Source changed before backup; retry after reviewing it')
        backups[str(path)] = str(backup)
    (backup_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    applied = []
    try:
        for path, before, after in plan:
            descriptor, name = tempfile.mkstemp(prefix='.sic-media-', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(after)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(name, stat.S_IMODE(path.stat().st_mode))
                if path.is_symlink() or path.read_bytes() != before:
                    raise ValueError('Current source changed during preparation; retry after reviewing it')
                os.replace(name, path)
                applied.append((path, before, after))
            finally:
                Path(name).unlink(missing_ok=True)
    except BaseException:
        for path, before, after in reversed(applied):
            if not path.is_symlink() and path.read_bytes() == after:
                shutil.copy2(backups[str(path)], path)
        raise
    result.update(applied=True, restart_required=True, backups=backups,
                  backup=str(backup_dir / 'webrtc.py') if str(target) in backups else None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    # An import in a separate process validates installed media dependencies;
    # importing this module does not initialize Firebase or load an AI model.
    python = args.repo.resolve() / '.venv' / 'bin' / 'python'
    if not python.is_file():
        parser.error('Existing Pi .venv/bin/python is required')
    subprocess.run([str(python), '-c',
                    'import runpy,sys; runpy.run_path(sys.argv[1])', str(args.source.resolve())],
                   cwd=args.repo, check=True)
    print(json.dumps(update(args.repo, args.source, apply=args.apply), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
