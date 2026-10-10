from datetime import datetime, timezone
from types import SimpleNamespace
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pi4_attendance.auth import build_admin_dependency
from pi4_attendance.storage import PiStore


def test_dashboard_history_is_paginated_read_only_and_filters_unknown(tmp_path):
    from pi4_attendance.dashboard_feed import add_dashboard_routes
    store = PiStore(tmp_path)
    with sqlite3.connect(store.database_path) as db:
        db.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)", ('person','Name','S1',b'private',1,.1,'model','active','2026-01-01','2026-01-01'))
        for index, moment in enumerate(['2026-10-10T16:59:59+00:00','2026-10-10T17:00:00+00:00','2026-10-11T10:00:00+00:00'],1):
            db.execute('INSERT INTO attendance VALUES (?,?,?,?,?,?,?,?)',(index,'session'+str(index),'person','Name',9000,'["sonix","jieli"]',moment,'pending'))
        for index, reason in enumerate(['similarity_below_threshold','detection_score_below_threshold','no_candidate']):
            db.execute('INSERT INTO recognition_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',('obs'+str(index),'rejected',None,None,'sonix','2026-10-11T00:00:00+00:00',0,0,0,0,'[]','model',reason,'2026-10-11T00:00:01+00:00'))
    runtime = SimpleNamespace(store=store, settings=SimpleNamespace(pi_id='pi',api_auth_mode='shared_token',admin_shared_token='fixture'),health=lambda: {'status':'ok','camera_status':[]})
    app=FastAPI(); add_dashboard_routes(app,runtime,build_admin_dependency(runtime.settings))
    with TestClient(app) as client:
        assert client.get('/api/dashboard').status_code==401
        auth={'Authorization':'Bearer fixture'}
        query={'start_date':'2026-10-11','end_date':'2026-10-11','limit':1}
        first=client.get('/api/dashboard',params=query,headers=auth).json()
        assert [x['attendance_id'] for x in first['attendance']]==[2]
        assert first['next_attendance_id']==2
        assert first['unknown'][0]['observation_id']=='obs0'
        assert first['next_unknown_rowid']==1
        assert 'embedding_int8' not in first['users'][0]
        second=client.get('/api/dashboard',params={**query,'after_attendance_id':2,'after_unknown_rowid':1},headers=auth).json()
        assert [x['attendance_id'] for x in second['attendance']]==[3]
        assert second['unknown'][0]['observation_id']=='obs2'
        assert second['next_attendance_id'] is None and second['next_unknown_rowid'] is None
        exhausted=client.get('/api/dashboard',params={**query,'after_attendance_id':-1,'after_unknown_rowid':-1},headers=auth).json()
        assert exhausted['attendance']==[] and exhausted['unknown']==[]
        assert client.get('/api/dashboard',params={'start_date':'2026-02-31'},headers=auth).status_code==422
    store.close()
