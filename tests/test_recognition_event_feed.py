from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import logging

import numpy as np
from fastapi.testclient import TestClient

from pi4_attendance.api import create_app
from pi4_attendance.models import RecognitionDecision
from pi4_attendance.storage import PiStore
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.preview import PreviewWindow
from pi4_attendance.tools.console import format_attendance


def decision(status='rejected', reason='no_candidate'):
    return RecognitionDecision('event-1',status,None,None,'sonix',datetime.now(timezone.utc),
        .2,.01,.9,.8,(0,0,20,20),'runtime-model',reason)


def test_event_feed_includes_unknown_without_creating_attendance(tmp_path):
    store=PiStore(tmp_path)
    try:
        store.record_decision(decision(),300)
        assert store.recent_recognitions()==[]
        rows=store.recent_recognition_events(10)
        assert len(rows)==1
        row=rows[0]
        assert row['display_status']=='unknown'
        assert row['name'] is None
        assert row['reason']=='no_candidate'
        assert row['model_version']=='runtime-model'
        assert row['camera_ids']==['sonix']
        assert store.recent_attendance()==[]
    finally: store.close()


def test_poor_quality_is_rejected_instead_of_declaring_unknown(tmp_path):
    store=PiStore(tmp_path)
    try:
        store.record_decision(decision(reason='quality_below_threshold'),300)
        assert store.recent_recognition_events()[0]['display_status']=='rejected'
    finally: store.close()


def test_console_displays_unknown_reason_and_model():
    text=format_attendance([{'occurred_at':'2026-10-10T11:00:00+00:00', 'name':None,
        'display_status':'unknown','status':'rejected','reason':'no_candidate',
        'model_version':'runtime-model','camera_ids':['sonix'],'similarity':.2}])
    assert 'UNKNOWN' in text
    assert 'no_candidate' in text
    assert 'runtime-model' in text
    assert 'None' not in text


def test_new_event_api_is_admin_protected():
    runtime=SimpleNamespace(settings=SimpleNamespace(api_cors_origins=(),api_auth_mode='shared_token',admin_shared_token='fixture'),
        recognition_events=lambda limit:[{'display_status':'unknown','model_version':'runtime-model'}])
    with TestClient(create_app(runtime)) as client:
        assert client.get('/api/recognition-events').status_code==401
        response=client.get('/api/recognition-events',headers={'Authorization':'Bearer fixture'})
        assert response.status_code==200
        assert response.json()[0]['display_status']=='unknown'


def test_toggle_recognition_logs_same_model_and_no_restart(caplog):
    runtime=AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.preview=PreviewWindow(('sonix',),30)
    runtime.engine=SimpleNamespace(model_version='runtime-model')
    with caplog.at_level(logging.INFO):
        runtime.set_recognition(True)
        runtime.set_recognition(False)
    assert runtime.engine.model_version=='runtime-model'
    assert 'recognition_enabled=True' in caplog.text
    assert 'recognition_enabled=False' in caplog.text
    assert 'model_changed=False' in caplog.text
    assert 'runtime-model' in caplog.text


def test_remote_control_requires_admin_without_weakening_loopback_control():
    runtime=SimpleNamespace(settings=SimpleNamespace(api_cors_origins=(),api_auth_mode='shared_token',admin_shared_token='fixture'),
        control_status=lambda:{'recognition_enabled':False}, set_recognition=lambda value:{'recognition_enabled':value})
    with TestClient(create_app(runtime), client=('198.51.100.20',1234)) as client:
        assert client.patch('/api/control',json={'recognition_enabled':True}).status_code==403
        assert client.patch('/api/control/remote',json={'recognition_enabled':True}).status_code==401
        response=client.patch('/api/control/remote',json={'recognition_enabled':True},headers={'Authorization':'Bearer fixture'})
        assert response.status_code==200
        assert response.json()['recognition_enabled'] is True
