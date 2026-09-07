import json
from pathlib import Path
import queue
import threading
import time

from pi4_attendance.telemetry import JsonlTelemetryWriter, ProcSampler, RollingMetrics, TelemetryMonitor


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_rolling_metrics_reports_thread_safe_counters_rates_and_percentiles() -> None:
    clock = Clock()
    metrics = RollingMetrics(window_seconds=10.0, clock=clock)

    threads = [threading.Thread(target=lambda: [metrics.increment("captured") for _ in range(100)]) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for value in (10.0, 20.0, 30.0, 40.0):
        metrics.observe("inference_ms", value)
    metrics.set_gauge("active_cameras", 3)
    clock.value = 10.0

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["captured"] == 400
    assert snapshot["gauges"]["active_cameras"] == 3
    assert snapshot["rates_per_second"]["captured"] == 40.0
    assert snapshot["distributions"]["inference_ms"] == {
        "samples": 4,
        "latest": 40.0,
        "p50": 25.0,
        "p95": 38.5,
        "p99": 39.7,
    }

    clock.value = 11.0
    assert metrics.snapshot()["rates_per_second"]["captured"] == 0.0


def test_rolling_metrics_prunes_while_recording_without_snapshots() -> None:
    clock = Clock()
    metrics = RollingMetrics(window_seconds=10.0, clock=clock)

    for second in range(101):
        clock.value = float(second)
        metrics.increment("captured")
        metrics.observe("inference_ms", float(second))

    assert len(metrics._events["captured"]) == 11
    assert len(metrics._observations["inference_ms"]) == 11


def test_startup_rate_uses_elapsed_time_before_window_is_full() -> None:
    clock = Clock()
    metrics = RollingMetrics(window_seconds=10.0, clock=clock)
    for _ in range(30):
        metrics.increment("captured")

    clock.value = 1.0

    assert metrics.snapshot()["rates_per_second"]["captured"] == 30.0


def _write_proc_stat(path: Path, *, total_user: int, total_system: int, idle: int, process_ticks: int) -> None:
    (path / "self").mkdir(exist_ok=True)
    (path / "stat").write_text(f"cpu  {total_user} 0 {total_system} {idle} 0 0 0 0 0 0\n", encoding="utf-8")
    fields = ["0"] * 50
    fields[0] = "S"
    fields[11] = str(process_ticks)
    fields[12] = "0"
    (path / "self" / "stat").write_text("123 (sic worker) " + " ".join(fields), encoding="utf-8")


def test_proc_sampler_reads_system_and_process_usage(tmp_path: Path) -> None:
    clock = Clock()
    _write_proc_stat(tmp_path, total_user=100, total_system=100, idle=800, process_ticks=100)
    (tmp_path / "meminfo").write_text("MemTotal: 1000000 kB\nMemAvailable: 250000 kB\n", encoding="utf-8")
    (tmp_path / "self" / "status").write_text("Name:\tsic\nVmRSS:\t125000 kB\n", encoding="utf-8")
    sampler = ProcSampler(proc_root=tmp_path, clock=clock, clock_ticks=100, cpu_count=4)

    first = sampler.sample()
    assert first["cpu_percent"] == 0.0
    assert first["ram_percent"] == 75.0
    assert first["ram_used_mb"] == 732.422
    assert first["ram_available_mb"] == 244.141
    assert first["process_cpu_percent"] == 0.0
    assert first["process_rss_mb"] == 122.07

    clock.value = 1.0
    _write_proc_stat(tmp_path, total_user=150, total_system=150, idle=900, process_ticks=150)
    second = sampler.sample()
    assert second["cpu_percent"] == 50.0
    assert second["process_cpu_percent"] == 12.5


def test_jsonl_writer_never_blocks_when_queue_is_full_and_drains_on_stop(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    writer = JsonlTelemetryWriter(path, queue_size=1, batch_size=8, flush_interval=60.0)

    assert writer.submit({"sequence": 1}) is True
    assert writer.submit({"sequence": 2}) is False
    assert writer.dropped == 1

    writer.start()
    writer.stop()

    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [{"sequence": 1}]
    assert not writer.is_alive()


def test_idle_writer_keeps_a_bounded_wait_after_flush_deadline(tmp_path: Path, monkeypatch) -> None:
    writer = JsonlTelemetryWriter(tmp_path / "telemetry.jsonl")
    time_values = iter((0.0, 10.0))
    monkeypatch.setattr("pi4_attendance.telemetry.time.monotonic", lambda: next(time_values))
    waits = []

    class EmptyQueue:
        def get(self, timeout):
            waits.append(timeout)
            writer._stop.set()
            raise queue.Empty

        def empty(self):
            return True

    writer._queue = EmptyQueue()
    writer._run()

    assert waits == [0.1]


def test_writer_reports_open_failure_and_rejects_more_samples(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    writer = JsonlTelemetryWriter(blocked / "telemetry.jsonl")

    writer.start()
    writer.stop()

    assert writer.error
    assert writer.submit({"sequence": 1}) is False
    assert writer.dropped == 1


def test_monitor_samples_in_background_and_stops_cleanly(tmp_path: Path) -> None:
    class Sampler:
        def sample(self) -> dict:
            return {"cpu_percent": 12.5}

    path = tmp_path / "monitor.jsonl"
    metrics = RollingMetrics(window_seconds=1.0)
    metrics.increment("processed")
    writer = JsonlTelemetryWriter(path, flush_interval=0.01)
    monitor = TelemetryMonitor(Sampler(), metrics, writer, interval=0.01, run_id="run-1")

    writer.start()
    monitor.start()
    deadline = time.monotonic() + 1.0
    while monitor.samples == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    monitor.stop()
    writer.stop()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["system"]["cpu_percent"] == 12.5
    assert rows[0]["runtime"]["counters"]["processed"] == 1
    assert rows[0]["type"] == "sample"
    assert rows[0]["run_id"] == "run-1"
    assert "timestamp" in rows[0]
    assert not monitor.is_alive()


def test_monitor_keeps_running_after_a_sampling_error(tmp_path: Path) -> None:
    class Sampler:
        calls = 0

        def sample(self) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary proc read failure")
            return {"cpu_percent": 25.0}

    sampler = Sampler()
    path = tmp_path / "monitor.jsonl"
    writer = JsonlTelemetryWriter(path, flush_interval=0.01)
    monitor = TelemetryMonitor(sampler, RollingMetrics(), writer, interval=0.01)

    writer.start()
    monitor.start()
    deadline = time.monotonic() + 1.0
    while sampler.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    monitor.stop()
    writer.stop()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sampler.calls >= 2
    assert rows[-1]["system"]["cpu_percent"] == 25.0
