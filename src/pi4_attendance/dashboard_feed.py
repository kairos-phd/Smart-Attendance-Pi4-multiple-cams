"""Read-only, paginated dashboard projection of the existing Pi SQLite store."""
from datetime import date, datetime, time, timedelta, timezone
import json
import sqlite3
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Query


def add_dashboard_routes(app, runtime, require_admin):
    @app.get('/api/dashboard')
    def dashboard(start_date: str | None = None, end_date: str | None = None,
                  after_attendance_id: int = Query(default=0, ge=-1),
                  after_unknown_rowid: int = Query(default=0, ge=-1),
                  limit: int = Query(default=500, ge=1, le=500),
                  _: str = Depends(require_admin)):
        try:
            start = date.fromisoformat(start_date) if start_date else None
            end = date.fromisoformat(end_date) if end_date else None
            if start and end and end < start:
                raise ValueError('End precedes start')
            zone = ZoneInfo('Asia/Ho_Chi_Minh')
            lower = datetime.combine(start, time.min, zone).astimezone(timezone.utc).isoformat() if start else None
            upper = datetime.combine(end + timedelta(days=1), time.min, zone).astimezone(timezone.utc).isoformat() if end else None
        except (ValueError, OverflowError) as exc:
            raise HTTPException(422, 'Invalid dashboard date range') from exc

        def page(connection, query, parameters, cursor_key):
            rows = [dict(row) for row in connection.execute(query, parameters).fetchall()]
            more = len(rows) > limit
            rows = rows[:limit]
            return rows, rows[-1][cursor_key] if more else None

        def date_filter(column):
            clauses, args = [], []
            if lower:
                clauses.append(f'julianday({column})>=julianday(?)'); args.append(lower)
            if upper:
                clauses.append(f'julianday({column})<julianday(?)'); args.append(upper)
            return ''.join(' AND '+clause for clause in clauses), args

        health = runtime.health()
        database = runtime.store.database_path.resolve()
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=2) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('BEGIN')
            users = [dict(row) for row in connection.execute(
                'SELECT user_id,name,student_id,status,model_version,created_at,updated_at FROM users ORDER BY name,user_id')]
            attendance, next_attendance = [], None
            if after_attendance_id >= 0:
                clause, args = date_filter('attendance.occurred_at')
                attendance, next_attendance = page(connection,
                    'SELECT attendance.attendance_id,attendance.session_id,attendance.user_id,attendance.name,'
                    'users.student_id,attendance.confidence_scaled,attendance.camera_ids_json,'
                    'attendance.occurred_at,attendance.sync_state FROM attendance '
                    'LEFT JOIN users ON users.user_id=attendance.user_id WHERE attendance.attendance_id>?'
                    +clause+' ORDER BY attendance.attendance_id LIMIT ?',
                    [after_attendance_id,*args,limit+1], 'attendance_id')
            unknown, next_unknown = [], None
            if after_unknown_rowid >= 0:
                clause, args = date_filter('observed_at')
                unknown, next_unknown = page(connection,
                    'SELECT rowid AS cursor,observation_id,observed_at AS occurred_at,camera_id,'
                    'similarity_scaled,reason,model_version FROM recognition_events WHERE rowid>? '
                    "AND status='rejected' AND reason IN ('no_candidate','similarity_below_threshold')"
                    +clause+' ORDER BY rowid LIMIT ?', [after_unknown_rowid,*args,limit+1], 'cursor')
        for row in attendance:
            row['camera_ids'] = json.loads(row.pop('camera_ids_json'))
            row['similarity'] = row.pop('confidence_scaled')/10000
        for row in unknown:
            row['similarity'] = row.pop('similarity_scaled')/10000
            row.pop('cursor')
        return {'pi_id':runtime.settings.pi_id,'fetched_at':datetime.now(timezone.utc).isoformat(),
                'health':health,'users':users,'attendance':attendance,'unknown':unknown,
                'next_attendance_id':next_attendance,'next_unknown_rowid':next_unknown}
