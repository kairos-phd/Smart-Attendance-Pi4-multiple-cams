from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv

from ..face_engine import FaceEngine
from ..model_identity import model_identity


def catalog_compatibility(database: Path, compatibility_key: str) -> dict:
    report = {'database_exists': database.exists(), 'active_users': 0, 'compatible_users': 0,
              'incompatible_users': 0, 'model_versions': {}}
    if not database.exists():
        return report
    # mode=ro preserves WAL visibility and never creates/migrates a database.
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=5) as db:
        rows = db.execute("SELECT model_version, COUNT(*) FROM users WHERE status='active' GROUP BY model_version").fetchall()
    versions = {version: count for version, count in rows}
    report.update(model_versions=versions, active_users=sum(versions.values()),
                  compatible_users=versions.get(compatibility_key, 0))
    report['incompatible_users'] = report['active_users'] - report['compatible_users']
    return report


def describe(engine: FaceEngine, data_root: Path) -> dict:
    files = []
    for filename in engine.model_info['files']:
        path = Path(filename)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    manifest_warnings = []
    pack_root = engine.model_root / 'models' / engine.model_pack
    manifest = pack_root / 'manifest.json'
    if manifest.exists():
        content = json.loads(manifest.read_text())
        for row in content.get('files', []):
            name = row.get('name') if isinstance(row, dict) else None
            if name and not (pack_root / name).is_file():
                manifest_warnings.append(f'Manifest references missing file: {name}')
    return {
        'model_version': engine.model_version, 'model_info': engine.model_info,
        'files': files, 'manifest_warnings': manifest_warnings,
        'legacy_pack_key': model_identity(engine.model_root, engine.model_pack).compatibility_key,
        'catalog': catalog_compatibility(data_root / 'smart_attendance.sqlite3', engine.model_version),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Read-only model contract and embedding compatibility preflight')
    parser.add_argument('--env', type=Path, default=Path.cwd() / '.env')
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--model-pack')
    parser.add_argument('--recognizer-model')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--det-size', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-compatible', action='store_true')
    args = parser.parse_args(argv)
    # Explicit service EnvironmentFile is authoritative over an unrelated SSH shell.
    load_dotenv(args.env, override=True)
    root = (args.model_root or Path(os.getenv('MODEL_ROOT', '.'))).expanduser().resolve()
    pack = args.model_pack or os.getenv('MODEL_PACK', 'pi4_fp32')
    if not list((root / 'models' / pack).glob('*.onnx')):
        parser.error('Configured model pack has no ONNX files; preflight will not download models')
    engine = FaceEngine(root, pack, args.det_size or int(os.getenv('FACE_DET_SIZE', '320')), .45, .06,
                        recognizer_model=args.recognizer_model or os.getenv('RECOGNIZER_MODEL', 'auto'))
    engine.start()
    data_root = (args.data_root or Path(os.getenv('DATA_ROOT', './data'))).expanduser().resolve()
    report = describe(engine, data_root)
    report.update(data_root=str(data_root), api_port=int(os.getenv('API_PORT', '8080')),
                  api_enabled=os.getenv('API_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'on'})
    if args.output:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, indent=2)
        os.chmod(args.output, 0o600)
    print(json.dumps(report, indent=2))
    return 2 if args.require_compatible and report['catalog']['incompatible_users'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
