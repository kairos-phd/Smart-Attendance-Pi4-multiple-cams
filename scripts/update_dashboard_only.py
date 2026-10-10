#!/usr/bin/env python3
"""Install the read-only dashboard route without replacing inference or model code."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


def patch_api(source):
    tree = ast.parse(source)
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == 'add_dashboard_routes' for node in ast.walk(tree)):
        return source
    factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_app']
    if len(factories) != 1:
        raise ValueError('Unknown API factory; source left unchanged')
    factory = factories[0]
    final = factory.body[-1]
    if (not factory.args.args or factory.args.args[0].arg != 'runtime'
            or not isinstance(final, ast.Return) or not isinstance(final.value, ast.Name)
            or final.value.id != 'app'
            or not any(isinstance(node, ast.Name) and node.id == 'require_admin' for node in ast.walk(factory))):
        raise ValueError('Unknown API factory; source left unchanged')
    lines = source.splitlines(keepends=True)
    lines.insert(final.lineno-1, '    from .dashboard_feed import add_dashboard_routes\n    add_dashboard_routes(app, runtime, require_admin)\n\n')
    result = ''.join(lines)
    compile(result, 'api.py', 'exec')
    return result


def update(repo, source, apply=False):
    root = repo.resolve(strict=True)/'src'/'pi4_attendance'
    api, feed = root/'api.py', root/'dashboard_feed.py'
    for path in (api, feed):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('Targets must be regular files')
    incoming = source.read_bytes()
    tree = ast.parse(incoming)
    if not any(isinstance(node, ast.FunctionDef) and node.name == 'add_dashboard_routes' for node in tree.body):
        raise ValueError('Candidate lacks dashboard factory')
    old_api = api.read_bytes()
    new_api = patch_api(old_api.decode()).encode()
    old_feed = feed.read_bytes() if feed.exists() else None
    # Install module first; bind its route only after it exists.
    plan = [(p,b,a) for p,b,a in [(feed,old_feed,incoming),(api,old_api,new_api)] if b != a]
    result = {'changed':bool(plan), 'applied':False, 'restart_required':False,
              'files':[str(p) for p,_,_ in plan], 'created_files':[str(p) for p,b,_ in plan if b is None]}
    if not apply or not plan:
        return result
    backup_root = repo.resolve().parent/'sic-dashboard-backups'
    backup_root.mkdir(mode=0o700,exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'),dir=backup_root))
    for path, before, after in plan:
        if before is not None:
            shutil.copy2(path,backup_dir/path.name)
            if (backup_dir/path.name).read_bytes() != before:
                raise ValueError('Source changed while backing up')
    result['backup_directory'] = str(backup_dir)
    (backup_dir/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    applied = []
    try:
        for path, before, after in plan:
            if path.is_symlink() or (path.read_bytes() if path.exists() else None) != before:
                raise ValueError('Source changed during preparation')
            fd, name = tempfile.mkstemp(prefix='.sic-dashboard-',dir=root)
            try:
                with os.fdopen(fd,'wb') as file:
                    file.write(after); file.flush(); os.fsync(file.fileno())
                os.chmod(name, stat.S_IMODE(path.stat().st_mode) if before is not None else 0o644)
                if path.is_symlink() or (path.read_bytes() if path.exists() else None) != before:
                    raise ValueError('Source changed during preparation')
                os.replace(name,path); applied.append((path,before,after))
            finally:
                Path(name).unlink(missing_ok=True)
    except BaseException:
        for path, before, after in reversed(applied):
            if not path.is_symlink() and path.read_bytes() == after:
                if before is None: path.unlink()
                else: shutil.copy2(backup_dir/path.name,path)
        raise
    result.update(applied=True,restart_required=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    print(json.dumps(update(args.repo,args.source,args.apply),indent=2))
