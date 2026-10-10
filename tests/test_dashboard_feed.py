from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import sqlite3
import pytest
from unittest.mock import patch

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


@pytest.fixture
def review_runtime(tmp_path):
    store = PiStore(tmp_path)
    with sqlite3.connect(store.database_path) as db:
        for identifier, status, reason in [('a'*32, 'rejected', 'no_candidate'),
                                           ('b'*32, 'rejected', 'detection_score_below_threshold'),
                                           ('c'*32, 'accepted', 'no_candidate'),
                                           ('d'*32, 'rejected', 'similarity_below_threshold')]:
            db.execute('INSERT INTO recognition_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identifier,status,None,'private candidate','sonix','2026-10-11T00:00:00+00:00',
                        0,0,0,0,'[]','model',reason,'2026-10-11T00:00:01+00:00'))
    runtime = SimpleNamespace(store=store, settings=SimpleNamespace(pi_id='pi',api_auth_mode='shared_token',admin_shared_token='fixture'),health=lambda: {'status':'ok'})
    yield runtime
    store.close()


def review_client(runtime):
    from pi4_attendance.dashboard_feed import add_dashboard_routes
    app = FastAPI()
    add_dashboard_routes(app, runtime, build_admin_dependency(runtime.settings))
    return TestClient(app)


def test_unknown_review_is_durable_and_does_not_change_source(review_runtime):
    runtime = review_runtime
    source_before = runtime.store.database_path.read_bytes()
    reviews = runtime.store.database_path.parent / 'dashboard_reviews.sqlite3'
    headers = {'Authorization':'Bearer fixture'}
    with review_client(runtime) as client:
        pending = client.get('/api/dashboard', headers=headers).json()['unknown']
        assert len(pending) == 2
        assert all(row['review_status'] == 'pending' and row['reviewed_at'] is None and row['reviewed_by'] is None for row in pending)
        assert not reviews.exists()
        assert all('name' not in row and 'candidate_name' not in row for row in pending)
        result = client.patch('/api/unknown-events/'+'a'*32+'/review', headers=headers, json={'decision':'reviewed'})
        assert result.status_code == 200
        assert result.json()['review_status'] == 'reviewed'
        assert result.json()['reviewed_by'] == 'shared-token-admin'
        assert datetime.fromisoformat(result.json()['reviewed_at']).tzinfo is not None
        assert result.json()['observation_id'] == 'a'*32
        assert reviews.stat().st_mode & 0o777 == 0o600
    with review_client(runtime) as client:
        rows = client.get('/api/dashboard', headers=headers).json()['unknown']
        assert rows[0]['review_status'] == 'reviewed'
        assert rows[1]['review_status'] == 'pending'
        result = client.patch('/api/unknown-events/'+'a'*32+'/review', headers=headers, json={'decision':'dismissed'})
        assert result.status_code == 200
        assert result.json()['review_status'] == 'dismissed'
    with review_client(runtime) as client:
        assert client.get('/api/dashboard', headers=headers).json()['unknown'][0]['review_status'] == 'dismissed'
    assert runtime.store.database_path.read_bytes() == source_before


def test_unknown_review_auth_validation_and_unknown_predicate(review_runtime):
    headers = {'Authorization':'Bearer fixture'}
    with review_client(review_runtime) as client:
        path = '/api/unknown-events/'+'a'*32+'/review'
        assert client.patch(path,json={'decision':'reviewed'}).status_code == 401
        for body in [{'decision':'pending'}, {'decision':True}, {'decision':'reviewed','reviewed_by':'victim'}]:
            assert client.patch(path,headers=headers,json=body).status_code == 422
        assert client.patch('/api/unknown-events/not-an-id/review',headers=headers,json={'decision':'reviewed'}).status_code == 422
        for identifier, code in [('b'*32,400), ('c'*32,400), ('e'*32,404)]:
            assert client.patch('/api/unknown-events/'+identifier+'/review',headers=headers,json={'decision':'reviewed'}).status_code == code
    assert not (review_runtime.store.database_path.parent/'dashboard_reviews.sqlite3').exists()


def test_unknown_reviews_reject_symlink_without_touching_target(review_runtime, tmp_path):
    target = tmp_path/'untouched'
    target.write_bytes(b'private')
    review_runtime.store.database_path.parent.joinpath('dashboard_reviews.sqlite3').symlink_to(target)
    with review_client(review_runtime) as client:
        headers = {'Authorization':'Bearer fixture'}
        assert client.patch('/api/unknown-events/'+'a'*32+'/review',headers=headers,json={'decision':'reviewed'}).status_code == 503
        assert client.get('/api/dashboard',headers=headers).status_code == 503
    assert target.read_bytes() == b'private'


def test_unknown_review_versions_follow_commit_order_when_clock_moves_backwards(review_runtime):
    from pi4_attendance import dashboard_feed
    source_before = review_runtime.store.database_path.read_bytes()
    moment = datetime(2026, 10, 10, tzinfo=timezone.utc)
    with review_client(review_runtime) as client, patch.object(dashboard_feed, 'datetime', wraps=datetime) as clock:
        clock.now.side_effect = [moment, moment + timedelta(microseconds=200), moment - timedelta(seconds=1), moment]
        path = '/api/unknown-events/'+'a'*32+'/review'
        versions = []
        for decision in ['reviewed', 'dismissed', 'reviewed', 'dismissed']:
            response = client.patch(path, headers={'Authorization':'Bearer fixture'}, json={'decision':decision})
            assert response.status_code == 200
            versions.append(datetime.fromisoformat(response.json()['reviewed_at']))
        assert all(later >= earlier + timedelta(milliseconds=1) for earlier, later in zip(versions, versions[1:]))
    with review_client(review_runtime) as client:
        persisted = client.get('/api/dashboard', headers={'Authorization':'Bearer fixture'}).json()['unknown'][0]
    assert persisted['review_status'] == 'dismissed'
    assert datetime.fromisoformat(persisted['reviewed_at']) == versions[-1]
    assert review_runtime.store.database_path.read_bytes() == source_before
