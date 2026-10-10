"""Dashboard projection and separate durable review metadata for unknown events."""
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import json
import os
import sqlite3
import stat
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict


class UnknownReviewUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['reviewed', 'dismissed']


def _review_connection(path, *, write=False):
    """Create only on writes, refusing symlinks and non-regular metadata files."""
    descriptor = None
    try:
        if write:
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        else:
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except FileNotFoundError:
                return None
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError('Review metadata must be a regular file')
        if write:
            os.fchmod(descriptor, 0o600)
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise OSError('Review metadata file changed')
        connection = sqlite3.connect(path.absolute().as_uri() + ('?mode=rw' if write else '?mode=ro'), uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, 'Unknown review storage is unavailable') from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unknown_reviews(path, identifiers):
    if not identifiers:
        return {}
    connection = _review_connection(path)
    if connection is None:
        return {}
    try:
        with closing(connection):
            # A concurrent first writer may have created the file before its first commit.
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='unknown_reviews'").fetchone():
                return {}
            marks = ','.join('?' for _ in identifiers)
            return {row['observation_id']: dict(row) for row in connection.execute(
                f'SELECT observation_id,review_status,reviewed_at,reviewed_by FROM unknown_reviews WHERE observation_id IN ({marks})', identifiers)}
    except sqlite3.Error as exc:
        raise HTTPException(503, 'Unknown review storage is unavailable') from exc


def add_dashboard_routes(app, runtime, require_admin):
    def reviews_path():
        return runtime.store.database_path.parent / 'dashboard_reviews.sqlite3'

    @app.patch('/api/unknown-events/{observation_id}/review')
    def review_unknown(update: UnknownReviewUpdate,
                       observation_id: str = Path(pattern=r'^[0-9a-f]{32}$'),
                       actor: str = Depends(require_admin)):
        database = runtime.store.database_path.resolve()
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=2)) as source:
            event = source.execute('SELECT status,reason FROM recognition_events WHERE observation_id=?', (observation_id,)).fetchone()
        if event is None:
            raise HTTPException(404, 'Unknown observation was not found')
        if event[0] != 'rejected' or event[1] not in {'no_candidate', 'similarity_below_threshold'}:
            raise HTTPException(400, 'Observation is not an unknown recognition event')
        connection = _review_connection(reviews_path(), write=True)
        try:
            with closing(connection), connection:
                # Serialize writers before choosing the version used by dashboard clients.
                connection.execute('BEGIN IMMEDIATE')
                connection.execute("CREATE TABLE IF NOT EXISTS unknown_reviews (observation_id TEXT PRIMARY KEY, review_status TEXT NOT NULL CHECK(review_status IN ('reviewed','dismissed')), reviewed_at TEXT NOT NULL, reviewed_by TEXT NOT NULL)")
                previous = connection.execute('SELECT reviewed_at FROM unknown_reviews WHERE observation_id=?', (observation_id,)).fetchone()
                reviewed_at = datetime.now(timezone.utc)
                if previous:
                    # Browsers compare timestamps at millisecond precision. Preserve ordering
                    # even if the clock moves backwards or two writes share a millisecond.
                    reviewed_at = max(reviewed_at, datetime.fromisoformat(previous['reviewed_at']) + timedelta(milliseconds=1))
                result = {'observation_id': observation_id, 'review_status': update.decision,
                          'reviewed_at': reviewed_at.isoformat(timespec='milliseconds'), 'reviewed_by': actor}
                connection.execute('INSERT INTO unknown_reviews (observation_id,review_status,reviewed_at,reviewed_by) VALUES (:observation_id,:review_status,:reviewed_at,:reviewed_by) ON CONFLICT(observation_id) DO UPDATE SET review_status=excluded.review_status, reviewed_at=excluded.reviewed_at, reviewed_by=excluded.reviewed_by', result)
        except sqlite3.Error as exc:
            raise HTTPException(503, 'Unknown review storage is unavailable') from exc
        return result

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
        reviews = _unknown_reviews(reviews_path(), [row['observation_id'] for row in unknown])
        for row in unknown:
            row['similarity'] = row.pop('similarity_scaled')/10000
            row.pop('cursor')
            row.update(reviews.get(row['observation_id'], {'review_status': 'pending', 'reviewed_at': None, 'reviewed_by': None}))
        return {'pi_id':runtime.settings.pi_id,'fetched_at':datetime.now(timezone.utc).isoformat(),
                'health':health,'users':users,'attendance':attendance,'unknown':unknown,
                'next_attendance_id':next_attendance,'next_unknown_rowid':next_unknown}
