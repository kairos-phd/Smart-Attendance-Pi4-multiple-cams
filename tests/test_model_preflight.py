import sqlite3
import pytest

from pi4_attendance.tools.model_info import catalog_compatibility


def test_preflight_preserves_database_and_counts_incompatible_active_users(tmp_path):
    path = tmp_path / 'catalog.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE users(model_version TEXT, status TEXT)')
        db.executemany('INSERT INTO users VALUES(?,?)', [('current','active'),('legacy','active'),('legacy','inactive')])
    before = path.read_bytes()
    report = catalog_compatibility(path, 'current')
    assert report == {'database_exists':True, 'active_users':2, 'compatible_users':1,
                      'incompatible_users':1, 'model_versions':{'current':1,'legacy':1}}
    assert path.read_bytes() == before


def test_preflight_missing_database_is_empty_without_creating_it(tmp_path):
    path = tmp_path / 'absent.sqlite3'
    assert catalog_compatibility(path, 'current')['active_users'] == 0
    assert not path.exists()


def test_preflight_does_not_hide_invalid_database(tmp_path):
    path = tmp_path / 'broken.sqlite3'; path.write_bytes(b'not sqlite')
    with pytest.raises(sqlite3.DatabaseError): catalog_compatibility(path, 'current')


def test_explicit_service_env_overrides_unrelated_shell_selectors(tmp_path, monkeypatch):
    import json
    from pi4_attendance.tools import model_info
    root = tmp_path/'models'; pack = root/'models'/'pack'; pack.mkdir(parents=True)
    (pack/'det.onnx').write_bytes(b'dummy')
    data = tmp_path/'actual-data'
    config = tmp_path/'.env'
    values = {'MODEL_ROOT':str(root),'MODEL_PACK':'pack','DATA_ROOT':str(data),
              'RECOGNIZER_MODEL':'pack','API_PORT':'8090','API_ENABLED':'true'}
    config.write_text('\n'.join(f'{k}={v}' for k,v in values.items()))
    for key in values: monkeypatch.setenv(key, 'unrelated-shell-value')
    class Engine:
        def __init__(self, model_root, model_pack, *a, **kw):
            assert model_root == root
            assert model_pack == 'pack'
            assert kw['recognizer_model'] == 'pack'
        def start(self): pass
    monkeypatch.setattr(model_info, 'FaceEngine', Engine)
    def describe(engine, data_root):
        assert data_root == data
        return {'catalog':{'incompatible_users':1}}
    monkeypatch.setattr(model_info, 'describe', describe)
    assert model_info.main(['--env',str(config),'--require-compatible']) == 2
