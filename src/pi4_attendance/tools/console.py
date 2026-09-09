from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable
from urllib import error, parse, request

from dotenv import dotenv_values


DEFAULT_URL = "http://127.0.0.1:8080"
REPOSITORY_ENV = Path(__file__).resolve().parents[3] / ".env"


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = request.build_opener(request.ProxyHandler({}), _NoRedirect())


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    token: str | None = None,
    timeout: float = 5.0,
):
    base_url = require_loopback_url(base_url)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    http_request = request.Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with _OPENER.open(http_request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON response: {exc}") from exc
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except (json.JSONDecodeError, AttributeError):
            detail = exc.reason
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Connection error: {reason}") from exc


def attendance_token(
    cli_token: str | None,
    environ: dict[str, str] | os._Environ[str] = os.environ,
    dotenv_path: Path = REPOSITORY_ENV,
) -> str:
    token = cli_token or environ.get("ADMIN_SHARED_TOKEN") or dotenv_values(dotenv_path).get("ADMIN_SHARED_TOKEN")
    if not token or not str(token).strip():
        raise ValueError("Attendance access requires --token, ADMIN_SHARED_TOKEN, or ADMIN_SHARED_TOKEN in repository .env")
    return str(token).strip()


def require_loopback_url(base_url: str) -> str:
    try:
        parsed = parse.urlsplit(base_url)
        host = parsed.hostname
        parsed.port
        loopback = host == "localhost" or (host is not None and ipaddress.ip_address(host).is_loopback)
    except (TypeError, ValueError):
        raise ValueError("URL must be a valid loopback HTTP(S) URL") from None
    if (
        base_url != base_url.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not loopback
    ):
        raise ValueError("URL must be a valid loopback HTTP(S) URL")
    return base_url


def format_control(status: dict) -> str:
    enrollment = status.get("enrollment") or {}
    online = ", ".join(status.get("online_camera_ids") or []) or "-"
    return "\n".join(
        (
            f"Runtime: {status.get('runtime_state', '-')}",
            f"Online cameras: {online}",
            f"Selected camera: {status.get('selected_camera_id', '-')}",
            f"Recognition: {'ON' if status.get('recognition_enabled') else 'OFF'}",
            f"Enrollment captured: {enrollment.get('captured', 0)}",
            f"Enrollment busy: {status.get('enrollment_busy', False)}",
            f"Enrollment status: {status.get('enrollment_status') or '-'}",
            "Keys: [A] recognition  [E] enroll  [1/2/3] camera  [C] capture  [Enter] submit  [X] cancel  [Q] quit",
        )
    )


def handle_control_key(
    key: str,
    status: dict,
    call: Callable[..., dict],
    prompt: Callable[[], tuple[str, str]] | None = None,
) -> bool:
    key = key.lower()
    if key == "q":
        return False
    if key == "a":
        call("/api/control", method="PATCH", payload={"recognition_enabled": not status.get("recognition_enabled", False)})
    elif key == "e" and prompt is not None:
        name, student_id = prompt()
        call("/api/control/enrollment/start", method="POST", payload={"name": name, "student_id": student_id})
    elif key in "123":
        cameras = status.get("camera_ids") or []
        index = int(key) - 1
        if index < len(cameras):
            call("/api/control", method="PATCH", payload={"camera_id": cameras[index]})
    elif key == "c":
        call("/api/control/enrollment/capture", method="POST")
    elif key in {"\n", "\r"}:
        call("/api/control/enrollment/submit", method="POST")
    elif key == "x":
        call("/api/control/enrollment", method="DELETE")
    return True


def _metric(value, suffix: str = "") -> str:
    return "-" if value is None else f"{value}{suffix}"


def _percentiles(distributions: dict, name: str) -> str:
    values = distributions.get(name) or {}
    return " / ".join(_metric(values.get(key)) for key in ("p50", "p95", "p99"))


def _display_width(text: str) -> int:
    return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def format_performance(health: dict) -> str:
    system = health.get("system") or {}
    performance = health.get("performance_metrics") or {}
    counters = performance.get("counters") or {}
    rates = performance.get("rates_per_second") or {}
    gauges = performance.get("gauges") or {}
    distributions = performance.get("distributions") or {}
    camera_rows = health.get("camera_status") or []
    camera_ids = sorted({row.get("camera_id") for row in camera_rows if row.get("camera_id")})
    online = ", ".join(
        row["camera_id"] for row in camera_rows if row.get("status") == "online" and row.get("camera_id")
    ) or "-"
    fields = [
        f"CPU: {_metric(system.get('cpu_percent'), '%')}",
        f"RAM: {_metric(system.get('ram_percent'), '%')}",
        f"Process RSS: {_metric(system.get('process_rss_mb'), ' MB')}",
        f"Online cameras: {online}",
    ]
    for camera_id in camera_ids:
        fields.append(
            f"{camera_id} capture/enqueue FPS: {_metric(rates.get(f'camera.{camera_id}.captured'))} / "
            f"{_metric(rates.get(f'camera.{camera_id}.enqueued'))}"
        )
    replacement_values = [value for key, value in counters.items() if key.endswith(".queue_replaced")]
    replacements = sum(replacement_values) if replacement_values else None
    queue_depth = health.get("queue_depth", gauges.get("queue.depth"))
    fields.extend(
        (
            f"Inference FPS: {_metric(rates.get('inference.processed'))}",
            f"Queue depth: {_metric(queue_depth)}",
            f"Total replacements: {_metric(replacements)}",
            f"Telemetry dropped samples: {_metric((health.get('telemetry') or {}).get('samples_dropped'))}",
            f"Inference ms p50/p95/p99: {_percentiles(distributions, 'inference_ms')}",
            f"Queue ms p50/p95/p99: {_percentiles(distributions, 'queue_age_ms')}",
            f"Full-processing ms p50/p95/p99: {_percentiles(distributions, 'full_processing_ms')}",
        )
    )
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    lines = []
    for field in fields:
        if lines and _display_width(lines[-1] + " | " + field) <= width:
            lines[-1] += " | " + field
        else:
            lines.append(field)
    return "\n".join(lines)


def format_attendance(rows: list[dict]) -> str:
    lines = ["Time | Name | Student ID | Camera | Similarity"]
    width, height = shutil.get_terminal_size(fallback=(120, 24))
    remaining = height - 1 - math.ceil(_display_width(lines[0]) / width)
    for row in sorted(rows, key=lambda row: row.get("occurred_at") or "", reverse=True):
        similarity = row.get("similarity")
        similarity_text = "-" if similarity is None else f"{float(similarity):.4f}"
        line = (
            f"{row.get('occurred_at', '-')} | {row.get('name', '-')} | {row.get('student_id', '-')} | "
            f"{','.join(row.get('camera_ids') or []) or '-'} | {similarity_text}"
        )
        wrap_width = max(1, width - int(any(unicodedata.east_asian_width(char) in {"W", "F"} for char in line)))
        if len(lines) == 1 and remaining > 0 and _display_width(line) > remaining * wrap_width:
            available = remaining * wrap_width - 1
            clipped = []
            for char in line:
                available -= _display_width(char)
                if available < 0:
                    break
                clipped.append(char)
            line = "".join(clipped) + "…"
        used = math.ceil(_display_width(line) / wrap_width)
        if used > remaining:
            break
        lines.append(line)
        remaining -= used
    return "\n".join(lines)


def _viewer(fetch: Callable[[], object], formatter: Callable[[object], str], interval: float, once: bool) -> int:
    while True:
        try:
            output = formatter(fetch())
        except RuntimeError as exc:
            output = f"Error: {exc}"
        if not once:
            print("\033[2J\033[H", end="")
        print(output, end="\n" if once else "", flush=True)
        if once:
            return 0
        time.sleep(interval)


def _positive_interval(value: str) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("interval must be finite and greater than zero")
    return interval


def _viewer_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--interval", type=_positive_interval, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser


def performance_main(argv: list[str] | None = None) -> int:
    parser = _viewer_parser("SIC Pi 4 performance console")
    args = parser.parse_args(argv)
    try:
        args.url = require_loopback_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    return _viewer(lambda: request_json(args.url, "/health"), format_performance, args.interval, args.once)


def attendance_main(argv: list[str] | None = None) -> int:
    parser = _viewer_parser("SIC Pi 4 attendance log")
    parser.add_argument("--token")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        args.url = require_loopback_url(args.url)
        token = attendance_token(args.token)
    except ValueError as exc:
        parser.error(str(exc))
    return _viewer(
        lambda: request_json(args.url, f"/api/attendance?limit={args.limit}", token=token),
        format_attendance,
        args.interval,
        args.once,
    )


def _control_screen(stdscr, base_url: str, curses_module) -> None:
    status: dict = {}
    message = ""

    def call(path: str, *, method: str = "GET", payload: dict | None = None):
        return request_json(base_url, path, method=method, payload=payload)

    def prompt() -> tuple[str, str]:
        curses_module.echo()
        stdscr.timeout(-1)
        height, _ = stdscr.getmaxyx()
        stdscr.addstr(height - 2, 0, "Name: ")
        name = stdscr.getstr().decode().strip()
        stdscr.addstr(height - 1, 0, "Student ID: ")
        student_id = stdscr.getstr().decode().strip()
        curses_module.noecho()
        stdscr.timeout(500)
        return name, student_id

    stdscr.timeout(500)
    while True:
        try:
            status = call("/api/control")
        except RuntimeError as exc:
            message = f"Error: {exc}"
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        for row, line in enumerate((format_control(status) + (f"\n{message}" if message else "")).splitlines()):
            if row < height:
                stdscr.addnstr(row, 0, line, max(0, width - 1))
        stdscr.refresh()
        message = ""
        try:
            key = stdscr.get_wch()
        except curses_module.error:
            continue
        if not isinstance(key, str):
            continue
        try:
            if not handle_control_key(key, status, call, prompt):
                return
        except RuntimeError as exc:
            message = f"Error: {exc}"


def control_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIC Pi 4 control console")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    try:
        base_url = require_loopback_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    import curses

    curses.wrapper(_control_screen, base_url, curses)
    return 0


def console_main() -> int:
    if shutil.which("tmux") is None:
        print("tmux is required to run sic-console", file=sys.stderr)
        return 1
    def tmux(*args: str, capture: bool = False):
        kwargs = {"capture_output": True, "text": True} if capture else {}
        result = subprocess.run(["tmux", *args], check=True, **kwargs)
        return result.stdout.strip() if capture else None

    commands = ("sic-pi4-control", "sic-pi4-performance", "sic-pi4-attendance-log")
    panes = {}
    window = None
    try:
        try:
            tmux("has-session", "-t", "=sic")
            exists = True
        except subprocess.CalledProcessError:
            exists = False
        if exists:
            rows = tmux("list-panes", "-s", "-t", "=sic", "-F", "#{window_name}\t#{window_id}\t#{pane_id}\t#{pane_title}\t#{pane_dead}", capture=True)
            for row in rows.splitlines():
                name, window_id, pane_id, title, dead = row.split("\t")
                if name == "sic-console" and title in commands and (window is None or window == window_id):
                    window = window_id
                    panes[title] = (pane_id, dead == "1")
        if not panes:
            creation = ("new-window", "-d", "-t", "=sic:") if exists else ("new-session", "-d", "-s", "sic")
            window, left = tmux(*creation, "-n", "sic-console", "-P", "-F", "#{window_id}\t#{pane_id}", "", capture=True).split("\t")
            panes[commands[0]] = (left, True)
        tmux("set-option", "-w", "-t", window, "remain-on-exit", "on")
        tmux("set-option", "-w", "-t", window, "automatic-rename", "off")
        for command in commands:
            if command not in panes:
                if command == commands[0]:
                    target = panes.get(commands[1], panes.get(commands[2]))[0]
                    direction = ("-h", "-b", "-f")
                elif command == commands[1] and commands[2] in panes:
                    target = panes[commands[2]][0]
                    direction = ("-v", "-b")
                else:
                    target = panes[commands[commands.index(command) - 1]][0]
                    direction = ("-h",) if command == commands[1] else ("-v",)
                pane_id = tmux("split-window", *direction, "-t", target, "-P", "-F", "#{pane_id}", "env", command, capture=True)
                panes[command] = (pane_id, False)
            pane_id, dead = panes[command]
            if dead:
                tmux("respawn-pane", "-k", "-t", pane_id, "env", command)
            tmux("select-pane", "-t", pane_id, "-T", command)
        tmux("select-window", "-t", window)
        tmux("select-pane", "-t", panes[commands[0]][0])
        tmux("attach-session", "-t", "=sic")
    except subprocess.CalledProcessError as exc:
        print(f"tmux command failed with exit code {exc.returncode}", file=sys.stderr)
        return 1
    return 0
