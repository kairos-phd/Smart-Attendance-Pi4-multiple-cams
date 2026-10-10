#!/usr/bin/env python3
"""Read-only event viewer for old and new Pi runtimes; never updates models/data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import sqlite3
import time
from urllib import request, parse


UNKNOWN_REASONS = {'no_candidate', 'similarity_below_threshold'}
class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = request.build_opener(request.ProxyHandler({}), NoRedirect())


def read_events(database: Path, cursor: int | None, limit: int = 20):
    if not database.is_file(): return [], cursor
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        db.row_factory=sqlite3.Row
        maximum=db.execute('SELECT COALESCE(MAX(rowid),0) FROM recognition_events').fetchone()[0]
        reset=cursor is not None and maximum < cursor
        if reset: cursor=None
        query='SELECT event.rowid AS cursor,event.*,users.student_id FROM recognition_events AS event LEFT JOIN users ON users.user_id=event.user_id '
        if cursor is None:
            rows=list(reversed(db.execute(query+'ORDER BY event.rowid DESC LIMIT ?', (limit,)).fetchall()))
        else:
            rows=db.execute(query+'WHERE event.rowid>? ORDER BY event.rowid LIMIT ?', (cursor,limit)).fetchall()
    result=[]
    if reset: result.append({'type':'log_cursor_reset','reason':'database restored or log rows removed'})
    for row in rows:
        accepted=row['status']=='accepted'
        result.append({'type':'recognition','observation_id':row['observation_id'],
            'occurred_at':row['observed_at'],'decided_at':row['decided_at'],
            'camera_id':row['camera_id'],'status':row['status'],
            'display_status':'accepted' if accepted else 'unknown' if row['reason'] in UNKNOWN_REASONS else 'rejected',
            'name':row['name'] if accepted else None,'student_id':row['student_id'] if accepted else None,
            'similarity':row['similarity_scaled']/10000,'reason':row['reason'],'model_version':row['model_version']})
    return result, rows[-1]['cursor'] if rows else maximum if cursor is None else cursor


def state_event(health: dict, control: dict, previous: dict | None):
    model=health.get('model_version')
    previous_model=None if previous is None else previous.get('model_version')
    return {'type':'recognition_state','logged_at':datetime.now(timezone.utc).isoformat(),
        'recognition_enabled':control.get('recognition_enabled'), 'model_version':model,
        'previous_model_version':previous_model,
        'model_changed':previous is not None and model!=previous_model,
        'inference_alive':health.get('inference_alive'),'queue_depth':health.get('queue_depth'),
        'camera_ids':control.get('camera_ids',[])}


def get_json(url: str):
    # Avoid redirects/proxy leakage from the local operational endpoints.
    with OPENER.open(url,timeout=2) as response: return json.load(response)


def config_values(path: Path):
    result={}
    if path.is_file():
        for line in path.read_text().splitlines():
            match=re.match(r'^\s*(?:export\s+)?(DATA_ROOT|API_PORT)\s*=\s*(.*?)\s*$',line)
            if match:
                result[match[1]]=match[2].split(' #',1)[0].strip().strip('\"\'')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/sic4b/Smart-Attendance-Pi4-multiple-cams'))
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--url')
    parser.add_argument('--interval',type=float,default=1)
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval<.2: parser.error('--interval must be finite and at least 0.2 seconds')
    if not 1<=args.limit<=500: parser.error('--limit must be between 1 and 500')
    repo=args.repo.expanduser().resolve(); config=config_values(repo/'.env')
    data=(args.data_root or Path(config.get('DATA_ROOT','./data'))).expanduser()
    data=(data if data.is_absolute() else repo/data).resolve()
    url=(args.url or f"http://127.0.0.1:{int(config.get('API_PORT','8080'))}").rstrip('/')
    parsed=parse.urlparse(url)
    if (parsed.scheme not in {'http','https'} or parsed.hostname not in {'localhost','127.0.0.1','::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {'','/'}):
        parser.error('--url must be a loopback API origin without credentials')
    database=data/'smart_attendance.sqlite3'; cursor=None; previous=None
    while True:
        errors=False
        try:
            health=get_json(url+'/health'); control=get_json(url+'/api/control')
            if not isinstance(health,dict) or not isinstance(control,dict): raise ValueError('Expected health/control objects')
            if previous is None or any(previous.get(k)!=value for k,value in (('model_version',health.get('model_version')),('recognition_enabled',control.get('recognition_enabled')))):
                print(json.dumps(state_event(health,control,previous),ensure_ascii=False),flush=True)
            previous={'model_version':health.get('model_version'),'recognition_enabled':control.get('recognition_enabled')}
        except (OSError,ValueError) as exc:
            errors=True
            print(json.dumps({'type':'diagnostic_error','component':'local_api','error':str(exc)}),flush=True)
        try:
            if not database.exists():
                errors=True
                print(json.dumps({'type':'diagnostic_error','component':'sqlite','error':'Database missing; check DATA_ROOT/service configuration'}),flush=True)
            else:
                rows,cursor=read_events(database,cursor,args.limit)
                for row in rows: print(json.dumps(row,ensure_ascii=False),flush=True)
        except (OSError,sqlite3.Error) as exc:
            errors=True
            print(json.dumps({'type':'diagnostic_error','component':'sqlite','error':str(exc)}),flush=True)
        if args.once: return int(errors)
        time.sleep(args.interval)


if __name__=='__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: pass
