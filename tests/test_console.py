from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest


ROOT = Path(__file__).parents[1]


def console_module():
    try:
        from pi4_attendance.tools import console
    except ModuleNotFoundError:
        pytest.fail("pi4_attendance.tools.console is missing")
    return console


class Response:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_request_json_sends_body_headers_token_and_timeout(monkeypatch) -> None:
    console = console_module()
    seen = {}

    def urlopen(request, timeout):
        seen.update(request=request, timeout=timeout)
        return Response({"ok": True})

    monkeypatch.setattr(console._OPENER, "open", urlopen)

    result = console.request_json(
        "http://127.0.0.1:8080", "/api/control", method="PATCH",
        payload={"recognition_enabled": True}, token="secret", timeout=2.5,
    )

    request = seen["request"]
    assert result == {"ok": True}
    assert request.full_url == "http://127.0.0.1:8080/api/control"
    assert request.method == "PATCH"
    assert json.loads(request.data) == {"recognition_enabled": True}
    assert request.headers["Content-type"] == "application/json"
    assert request.headers["Authorization"] == "Bearer secret"
    assert seen["timeout"] == 2.5


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (HTTPError("http://local", 400, "Bad Request", {}, io.BytesIO(b'{"detail":"bad camera"}')), "HTTP 400: bad camera"),
        (URLError("connection refused"), "Connection error: connection refused"),
    ],
)
def test_request_json_reports_readable_errors(monkeypatch, error, message) -> None:
    console = console_module()
    monkeypatch.setattr(console._OPENER, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match=message):
        console.request_json("http://127.0.0.1:8080", "/health")


def test_request_json_reports_malformed_success(monkeypatch) -> None:
    console = console_module()
    response = Response(None)
    response.read = lambda: b"<html>service unavailable</html>"
    monkeypatch.setattr(console._OPENER, "open", lambda *_args, **_kwargs: response)
    with pytest.raises(RuntimeError, match="Invalid JSON response"):
        console.request_json("http://127.0.0.1:8080", "/health")


@pytest.mark.parametrize("interval", ["-1", "0", "nan", "inf", "-inf"])
@pytest.mark.parametrize("command", ["performance_main", "attendance_main"])
def test_viewer_rejects_invalid_interval_before_fetch(monkeypatch, command, interval) -> None:
    console = console_module()
    monkeypatch.setattr(console, "_viewer", lambda *_args: pytest.fail("invalid interval reached viewer"))
    with pytest.raises(SystemExit) as exc:
        extra = ["--token", "secret"] if command == "attendance_main" else []
        getattr(console, command)([f"--interval={interval}", *extra])
    assert exc.value.code == 2


def test_request_json_does_not_follow_redirects() -> None:
    console = console_module()
    sink_hit = threading.Event()

    class SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            sink_hit.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args) -> None:
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/sink")
            self.end_headers()

        def log_message(self, *_args) -> None:
            pass

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (sink, redirect)]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(RuntimeError, match="HTTP 302"):
            console.request_json(f"http://127.0.0.1:{redirect.server_port}", "/start", token="secret")
        assert not sink_hit.is_set()
    finally:
        redirect.shutdown()
        sink.shutdown()
        redirect.server_close()
        sink.server_close()


def test_request_opener_ignores_environment_proxies(monkeypatch) -> None:
    console = console_module()
    proxy_variables = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    )
    with monkeypatch.context() as environment:
        for name in proxy_variables:
            environment.delenv(name, raising=False)
        environment.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        console = importlib.reload(console)
        proxy_handlers = [
            handler for handler in console._OPENER.handlers
            if isinstance(handler, console.request.ProxyHandler)
        ]
        assert proxy_handlers == []
    importlib.reload(console)


def test_attendance_token_lookup_order(tmp_path, monkeypatch) -> None:
    console = console_module()
    dotenv = tmp_path / ".env"
    dotenv.write_text("ADMIN_SHARED_TOKEN=file-token\n", encoding="utf-8")

    assert console.attendance_token("cli-token", {"ADMIN_SHARED_TOKEN": "env-token"}, dotenv) == "cli-token"
    assert console.attendance_token(None, {"ADMIN_SHARED_TOKEN": "env-token"}, dotenv) == "env-token"
    assert console.attendance_token(None, {}, dotenv) == "file-token"
    dotenv.unlink()
    with pytest.raises(ValueError, match=r"--token.*ADMIN_SHARED_TOKEN.*\.env"):
        console.attendance_token(None, {}, dotenv)


def test_every_control_key_calls_the_expected_route() -> None:
    console = console_module()
    status = {
        "camera_ids": ["camera-a", "camera-b", "camera-c"],
        "recognition_enabled": True,
    }
    calls = []

    def call(path, *, method="GET", payload=None):
        calls.append((path, method, payload))
        return status

    assert console.handle_control_key("a", status, call) is True
    assert calls.pop() == ("/api/control", "PATCH", {"recognition_enabled": False})
    assert console.handle_control_key("e", status, call, lambda: ("Dang", "S001")) is True
    assert calls.pop() == ("/api/control/enrollment/start", "POST", {"name": "Dang", "student_id": "S001"})
    for key, camera_id in zip("123", status["camera_ids"], strict=True):
        assert console.handle_control_key(key, status, call) is True
        assert calls.pop() == ("/api/control", "PATCH", {"camera_id": camera_id})
    assert console.handle_control_key("c", status, call) is True
    assert calls.pop() == ("/api/control/enrollment/capture", "POST", None)
    assert console.handle_control_key("\n", status, call) is True
    assert calls.pop() == ("/api/control/enrollment/submit", "POST", None)
    assert console.handle_control_key("x", status, call) is True
    assert calls.pop() == ("/api/control/enrollment", "DELETE", None)
    assert console.handle_control_key("q", status, call) is False
    assert calls == []


def test_control_key_ignores_unconfigured_camera_and_unknown_key() -> None:
    console = console_module()
    calls = []
    call = lambda *args, **kwargs: calls.append((args, kwargs))

    assert console.handle_control_key("3", {"camera_ids": ["camera-1"]}, call) is True
    assert console.handle_control_key("?", {"camera_ids": ["camera-1"]}, call) is True
    assert calls == []


def test_control_format_shows_required_state() -> None:
    console = console_module()
    output = console.format_control({
        "runtime_state": "running",
        "online_camera_ids": ["camera-1", "camera-3"],
        "selected_camera_id": "camera-1",
        "recognition_enabled": True,
        "enrollment": {"captured": 3},
        "enrollment_busy": False,
        "enrollment_status": "Captured 3/8",
    })

    for expected in ("running", "camera-1, camera-3", "camera-1", "ON", "3", "False", "Captured 3/8"):
        assert expected in output


def test_performance_format_shows_all_metrics_and_missing_values() -> None:
    console = console_module()
    health = {
        "system": {"cpu_percent": 12.5, "ram_percent": 44.0, "process_rss_mb": 321.5},
        "camera_status": [{"camera_id": "camera-1", "status": "online"}],
        "queue_depth": 2,
        "performance_metrics": {
            "counters": {"camera.camera-1.queue_replaced": 7},
            "rates_per_second": {
                "camera.camera-1.captured": 9.1,
                "camera.camera-1.enqueued": 8.2,
                "inference.processed": 4.3,
            },
            "distributions": {
                "inference_ms": {"p50": 10, "p95": 20, "p99": 30},
                "queue_age_ms": {"p50": 1, "p95": 2, "p99": 3},
            },
        },
        "telemetry": {"samples_dropped": 5},
    }

    output = console.format_performance(health)

    for expected in (
        "CPU: 12.5%", "RAM: 44.0%", "Process RSS: 321.5 MB", "Online cameras: camera-1",
        "camera-1 capture/enqueue FPS: 9.1 / 8.2", "Inference FPS: 4.3", "Queue depth: 2",
        "Total replacements: 7", "Inference ms p50/p95/p99: 10 / 20 / 30",
        "Queue ms p50/p95/p99: 1 / 2 / 3", "Full-processing ms p50/p95/p99: - / - / -",
        "Telemetry dropped samples: 5",
    ):
        assert expected in output

    missing = console.format_performance({})
    assert "CPU: -" in missing
    assert "Inference ms p50/p95/p99: - / - / -" in missing
    assert "Total replacements: -" in missing
    assert "Telemetry dropped samples: -" in missing


def test_attendance_format_has_heading_and_decimal_similarity() -> None:
    console = console_module()
    output = console.format_attendance([{
        "occurred_at": "2026-09-08T10:11:12+00:00",
        "name": "Dang",
        "student_id": "S001",
        "camera_ids": ["camera-1", "camera-2"],
        "similarity": 0.8123,
    }])

    assert output.splitlines()[0] == "Time | Name | Student ID | Camera | Similarity"
    assert "2026-09-08T10:11:12+00:00 | Dang | S001 | camera-1,camera-2 | 0.8123" in output
    assert "%" not in output


def test_viewers_fit_small_panes_and_keep_newest_attendance(monkeypatch) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback: os.terminal_size((120, 10)))
    health = {
        "system": {"cpu_percent": 12, "ram_percent": 34},
        "camera_status": [{"camera_id": f"camera-{i}", "status": "online"} for i in range(1, 4)],
    }
    performance = console.format_performance(health).splitlines()
    assert len(performance) <= 10
    assert "CPU: 12%" in performance[0] and "RAM: 34%" in performance[0]


def test_attendance_caps_rows_to_pane_height(monkeypatch) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback: os.terminal_size((120, 10)))
    rows = [{"occurred_at": f"2026-09-08T10:00:{i:02d}", "name": f"person-{i:02d}"} for i in range(30)]
    attendance = console.format_attendance(rows).splitlines()
    assert len(attendance) <= 9
    assert attendance[0].startswith("Time | Name")
    assert "person-29" in attendance[1]
    assert "person-00" not in "\n".join(attendance)


def test_performance_fits_physical_rows_in_narrow_pane(monkeypatch) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback: os.terminal_size((59, 10)))
    health = {
        "system": {"cpu_percent": 12.5, "ram_percent": 44.0, "process_rss_mb": 321.5},
        "camera_status": [{"camera_id": f"camera-{i}", "status": "online"} for i in range(1, 4)],
        "queue_depth": 2,
        "performance_metrics": {
            "rates_per_second": {"inference.processed": 4.3, **{
                f"camera.camera-{i}.{event}": rate
                for i in range(1, 4) for event, rate in (("captured", 9.1), ("enqueued", 8.2))
            }},
            "counters": {"camera.camera-1.queue_replaced": 7},
            "distributions": {
                name: {"p50": 10, "p95": 20, "p99": 30}
                for name in ("inference_ms", "queue_age_ms", "full_processing_ms")
            },
        },
        "telemetry": {"samples_dropped": 5},
    }
    output = console.format_performance(health)
    assert sum(max(1, (len(line) + 58) // 59) for line in output.splitlines()) <= 10
    for value in (
        "CPU: 12.5%", "RAM: 44.0%", "Process RSS: 321.5 MB", "Inference FPS: 4.3",
        "Queue depth: 2", "Total replacements: 7", "Telemetry dropped samples: 5",
        "Inference ms p50/p95/p99: 10 / 20 / 30", "Queue ms p50/p95/p99: 10 / 20 / 30",
        "Full-processing ms p50/p95/p99: 10 / 20 / 30",
    ):
        assert value in output
    for i in range(1, 4):
        assert f"camera-{i} capture/enqueue FPS: 9.1 / 8.2" in output


def test_attendance_caps_physical_rows_in_narrow_pane(monkeypatch) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback: os.terminal_size((59, 10)))
    rows = [{
        "occurred_at": f"2026-09-08T10:00:{i:02d}+00:00", "name": "Nguyen Van A",
        "student_id": f"S{i:03d}", "camera_ids": ["camera-1"], "similarity": 0.8123,
    } for i in range(8)]
    output = console.format_attendance(rows)
    assert sum(max(1, (len(line) + 58) // 59) for line in output.splitlines()) <= 10
    assert output.startswith("Time | Name | Student ID | Camera | Similarity")
    assert "10:00:07+00:00 | Nguyen Van A | S007 | camera-1 | 0.8123" in output
    assert "S000" not in output


def test_attendance_keeps_oversized_newest_row_with_wide_characters_visible(monkeypatch) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback: os.terminal_size((59, 10)))
    rows = [
        {"occurred_at": "2026-09-08T10:00:07+00:00", "name": "界" * 1000},
        {"occurred_at": "2026-09-08T10:00:06+00:00", "name": "Old"},
    ]
    output = console.format_attendance(rows)
    physical_rows = 0
    for line in output.splitlines():
        occupied = 0
        physical_rows += 1
        for char in line:
            cells = 2 if char == "界" else 1
            if occupied + cells > 59:
                physical_rows += 1
                occupied = 0
            occupied += cells
    assert physical_rows <= 9
    assert "10:00:07+00:00 | 界" in output
    assert output.endswith("…")
    assert "Old" not in output


def test_live_viewer_does_not_scroll_header_off_full_height_pane(monkeypatch, capsys) -> None:
    console = console_module()
    output = "CPU: 12% | RAM: 34%\n" + "\n".join(["metrics"] * 9)
    monkeypatch.setattr(console.time, "sleep", lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        console._viewer(lambda: {}, lambda _: output, 1.0, False)
    assert capsys.readouterr().out == "\033[2J\033[H" + output


def test_entry_points_are_exact() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts | {
        "sic-pi4-control": "pi4_attendance.tools.console:control_main",
        "sic-pi4-performance": "pi4_attendance.tools.console:performance_main",
        "sic-pi4-attendance-log": "pi4_attendance.tools.console:attendance_main",
        "sic-console": "pi4_attendance.tools.console:console_main",
    } == scripts


def test_install_script_installs_console_commands_and_tmux() -> None:
    script = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    apt_command = script[script.index("sudo apt install"):script.index("\n\n", script.index("sudo apt install"))]
    assert "tmux" in apt_command.replace("\\\n", " ").split()
    assert [line for line in script.splitlines() if line.startswith("sudo ln -sfn ")] == [
        f'sudo ln -sfn "$ROOT/.venv/bin/{command}" "/usr/local/bin/{command}"'
        for command in (
            "sic-pi4-control",
            "sic-pi4-performance",
            "sic-pi4-attendance-log",
            "sic-console",
        )
    ]


@pytest.mark.parametrize("missing", [None, "sic-pi4-control", "sic-pi4-performance", "sic-pi4-attendance-log"])
@pytest.mark.parametrize("dead", [False, True])
def test_console_repairs_named_clients_in_existing_tmux_session(monkeypatch, missing, dead) -> None:
    console = console_module()
    calls = []
    monkeypatch.setattr(console.shutil, "which", lambda name: "/usr/bin/tmux")
    commands = ("sic-pi4-control", "sic-pi4-performance", "sic-pi4-attendance-log")
    panes = {f"%{i + 7}": [name, dead] for i, name in enumerate(commands) if name != missing}

    def run(argv, **kwargs):
        calls.append(argv)
        action = argv[1]
        if action == "list-panes":
            return SimpleNamespace(stdout="\n".join(f"sic-console\t@4\t{pane}\t{name}\t{int(stopped)}" for pane, (name, stopped) in panes.items()))
        if action == "respawn-pane":
            panes[argv[argv.index("-t") + 1]] = [argv[-1], False]
        if action == "split-window":
            panes["%30"] = [argv[-1], False]
            return SimpleNamespace(stdout="%30\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(console.subprocess, "run", run)
    assert console.console_main() == 0
    assert calls[0] == ["tmux", "has-session", "-t", "=sic"]
    assert calls[-1] == ["tmux", "attach-session", "-t", "=sic"]
    assert sorted(name for name, stopped in panes.values() if not stopped) == sorted(commands)
    assert any("remain-on-exit" in call and "on" in call for call in calls)
    assert not any(call[1].startswith("kill-") for call in calls)
    if missing == "sic-pi4-control":
        assert any(call[1] == "split-window" and "-h" in call and "-b" in call and "-f" in call for call in calls)


def test_console_creates_three_panes_when_session_is_missing(monkeypatch) -> None:
    console = console_module()
    calls = []
    monkeypatch.setattr(console.shutil, "which", lambda name: "/usr/bin/tmux")

    pane_ids = iter(("@4\t%7\n", "%12\n", "%20\n"))

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "has-session":
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout=next(pane_ids, "") if argv[1] in {"new-session", "split-window"} else "")

    monkeypatch.setattr(console.subprocess, "run", run)

    console.console_main()

    argv_calls = [argv for argv, _ in calls]
    assert argv_calls[0] == ["tmux", "has-session", "-t", "=sic"]
    assert next(argv for argv in argv_calls if argv[1] == "new-session")[-1] == ""
    assert [argv[-1] for argv in argv_calls if argv[1] in {"respawn-pane", "split-window"}] == [
        "sic-pi4-control", "sic-pi4-performance", "sic-pi4-attendance-log",
    ]
    retention_index = next(i for i, argv in enumerate(argv_calls) if "remain-on-exit" in argv)
    assert all(i > retention_index for i, argv in enumerate(argv_calls) if argv[1] in {"respawn-pane", "split-window"})
    assert all(argv[-2] == "env" for argv in argv_calls if argv[1] in {"respawn-pane", "split-window"})
    assert [argv[argv.index("-t") + 1] for argv in argv_calls if argv[1] == "split-window"] == ["%7", "%12"]
    assert [argv[-1] for argv in argv_calls if argv[1] == "select-pane" and "-T" in argv] == [
        "sic-pi4-control", "sic-pi4-performance", "sic-pi4-attendance-log",
    ]
    assert any("remain-on-exit" in argv and "@4" in argv for argv in argv_calls)
    assert argv_calls[-1] == ["tmux", "attach-session", "-t", "=sic"]


def test_console_reports_missing_tmux_and_command_errors(monkeypatch, capsys) -> None:
    console = console_module()
    monkeypatch.setattr(console.shutil, "which", lambda name: None)
    assert console.console_main() == 1
    assert "tmux is required" in capsys.readouterr().err

    monkeypatch.setattr(console.shutil, "which", lambda name: "/usr/bin/tmux")

    def fail(argv, **_kwargs):
        if argv[1] == "has-session":
            raise subprocess.CalledProcessError(1, argv)
        raise subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(console.subprocess, "run", fail)
    assert console.console_main() == 1
    assert "tmux command failed" in capsys.readouterr().err


def test_control_url_must_be_loopback() -> None:
    console = console_module()
    assert console.require_loopback_url("http://localhost:8080") == "http://localhost:8080"
    assert console.require_loopback_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    with pytest.raises(ValueError, match="loopback"):
        console.require_loopback_url("http://192.168.1.10:8080")


@pytest.mark.parametrize(
    "url",
    (
        "ftp://127.0.0.1:8080",
        "http://user:password@127.0.0.1:8080",
        "http://[::1",
        "http://127.0.0.1:bad",
        "http:///missing-host",
    ),
)
def test_control_url_rejects_non_http_userinfo_and_malformed_urls(url) -> None:
    console = console_module()
    with pytest.raises(ValueError, match="loopback HTTP"):
        console.require_loopback_url(url)


@pytest.mark.parametrize(
    ("command", "extra"),
    (("control_main", []), ("performance_main", ["--once"]), ("attendance_main", ["--once", "--token", "secret"])),
)
def test_all_clients_validate_url_before_running(monkeypatch, command, extra) -> None:
    console = console_module()
    monkeypatch.setattr(console, "_viewer", lambda *_args: pytest.fail("viewer ran before URL validation"))
    with pytest.raises(SystemExit, match="2"):
        getattr(console, command)(["--url", "https://example.com", *extra])


def test_control_screen_preserves_action_error_until_next_render(monkeypatch) -> None:
    console = console_module()
    status = {
        "camera_ids": ["camera-1"],
        "recognition_enabled": False,
        "runtime_state": "running",
    }

    def request_json(_base_url, path, *, method="GET", payload=None):
        if path.endswith("capture"):
            raise RuntimeError("Wait for a new camera frame")
        return status

    monkeypatch.setattr(console, "request_json", request_json)

    class FakeScreen:
        keys = iter(("c", "q"))

        def __init__(self) -> None:
            self.current = []
            self.frames = []

        def timeout(self, _value) -> None:
            pass

        def erase(self) -> None:
            self.current = []

        def getmaxyx(self):
            return 20, 120

        def addnstr(self, _row, _column, value, _limit) -> None:
            self.current.append(value)

        def refresh(self) -> None:
            self.frames.append("\n".join(self.current))

        def get_wch(self):
            return next(self.keys)

    screen = FakeScreen()
    curses_module = SimpleNamespace(error=RuntimeError, echo=lambda: None, noecho=lambda: None)

    console._control_screen(screen, "http://127.0.0.1:8080", curses_module)

    assert "Error: Wait for a new camera frame" in screen.frames[1]


def test_console_module_does_not_import_curses_at_import_time() -> None:
    console = console_module()
    assert "curses" not in console.__dict__
    assert "curses" not in sys.modules or sys.platform != "win32"
