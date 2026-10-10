import importlib.util
from pathlib import Path
import sys

from pi4_attendance.storage import PiStore
from test_recognition_event_feed import decision


def watcher():
    path=Path(__file__).parents[1]/'scripts'/'watch_recognition.py'
    spec=importlib.util.spec_from_file_location('watch_recognition',path)
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
    return m


def test_readonly_watcher_includes_unknown_and_does_not_repeat(tmp_path):
    m=watcher(); store=PiStore(tmp_path)
    try:
        store.record_decision(decision(),300)
        rows,cursor=m.read_events(store.database_path,None,10)
        assert len(rows)==1
        assert rows[0]['display_status']=='unknown'
        assert rows[0]['model_version']=='runtime-model'
        assert m.read_events(store.database_path,cursor,10)[0]==[]
        assert store.recent_attendance()==[]
    finally: store.close()


def test_watcher_reports_model_change_separately_from_toggle():
    m=watcher()
    previous={'recognition_enabled':False,'model_version':'model-A'}
    row=m.state_event({'model_version':'model-A'}, {'recognition_enabled':True}, previous)
    assert row['model_changed'] is False
    assert row['recognition_enabled'] is True
    changed=m.state_event({'model_version':'model-B'}, {'recognition_enabled':True},previous)
    assert changed['model_changed'] is True
    assert changed['previous_model_version']=='model-A'


def test_missing_database_does_not_get_created(tmp_path):
    m=watcher(); path=tmp_path/'absent.sqlite3'
    rows,_=m.read_events(path,None,10)
    assert rows==[]
    assert not path.exists()
