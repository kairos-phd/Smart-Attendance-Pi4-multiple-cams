from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Callable


class RollingMetrics:
    def __init__(self, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._started = clock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._events: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._observations: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def increment(self, name: str, amount: float = 1.0) -> None:
        now = self._clock()
        with self._lock:
            self._counters[name] += amount
            self._events[name].append((now, amount))
            self._prune(self._events[name], now - self.window_seconds)

    def observe(self, name: str, value: float) -> None:
        now = self._clock()
        with self._lock:
            self._observations[name].append((now, float(value)))
            self._prune(self._observations[name], now - self.window_seconds)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def snapshot(self) -> dict:
        now = self._clock()
        cutoff = now - self.window_seconds
        elapsed = min(self.window_seconds, max(0.0, now - self._started))
        with self._lock:
            rates = {}
            for name, events in self._events.items():
                self._prune(events, cutoff)
                rates[name] = round(sum(amount for _, amount in events) / elapsed, 3) if elapsed else 0.0

            distributions = {}
            for name, observations in self._observations.items():
                self._prune(observations, cutoff)
                if observations:
                    values = sorted(value for _, value in observations)
                    distributions[name] = {
                        "samples": len(values),
                        "latest": round(observations[-1][1], 3),
                        "p50": round(self._percentile(values, 0.50), 3),
                        "p95": round(self._percentile(values, 0.95), 3),
                        "p99": round(self._percentile(values, 0.99), 3),
                    }

            return {
                "counters": {name: self._number(value) for name, value in self._counters.items()},
                "gauges": {name: self._number(value) for name, value in self._gauges.items()},
                "rates_per_second": rates,
                "distributions": distributions,
            }

    @staticmethod
    def _prune(values: deque[tuple[float, float]], cutoff: float) -> None:
        while values and values[0][0] < cutoff:
            values.popleft()

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    @staticmethod
    def _number(value: float) -> int | float:
        return int(value) if value.is_integer() else round(value, 3)


class ProcSampler:
    def __init__(
        self,
        proc_root: Path = Path("/proc"),
        clock: Callable[[], float] = time.monotonic,
        clock_ticks: int | None = None,
        cpu_count: int | None = None,
    ) -> None:
        self.proc_root = Path(proc_root)
        self._clock = clock
        self._clock_ticks = clock_ticks or (
            int(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100
        )
        self._cpu_count = max(1, cpu_count or os.cpu_count() or 1)
        self._previous_system: tuple[int, int] | None = None
        self._previous_process: tuple[float, int] | None = None

    def sample(self) -> dict:
        now = self._clock()
        total_ticks, idle_ticks = self._system_ticks()
        process_ticks = self._process_ticks()
        total_memory_kb, available_memory_kb = self._memory_kb()
        rss_kb = self._process_rss_kb()

        cpu_percent = 0.0
        if self._previous_system is not None:
            previous_total, previous_idle = self._previous_system
            total_delta = total_ticks - previous_total
            idle_delta = idle_ticks - previous_idle
            if total_delta > 0:
                cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta

        process_cpu_percent = 0.0
        if self._previous_process is not None:
            previous_time, previous_ticks = self._previous_process
            elapsed = now - previous_time
            if elapsed > 0:
                process_cpu_percent = (
                    100.0 * (process_ticks - previous_ticks) / self._clock_ticks / elapsed / self._cpu_count
                )

        self._previous_system = total_ticks, idle_ticks
        self._previous_process = now, process_ticks
        used_memory_kb = max(0, total_memory_kb - available_memory_kb)
        return {
            "cpu_percent": round(max(0.0, cpu_percent), 3),
            "ram_percent": round(100.0 * used_memory_kb / total_memory_kb, 3) if total_memory_kb else 0.0,
            "ram_used_mb": round(used_memory_kb / 1024.0, 3),
            "ram_available_mb": round(available_memory_kb / 1024.0, 3),
            "process_cpu_percent": round(max(0.0, process_cpu_percent), 3),
            "process_rss_mb": round(rss_kb / 1024.0, 3),
        }

    def _system_ticks(self) -> tuple[int, int]:
        fields = (self.proc_root / "stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        ticks = [int(value) for value in fields]
        return sum(ticks), ticks[3] + (ticks[4] if len(ticks) > 4 else 0)

    def _process_ticks(self) -> int:
        fields = (self.proc_root / "self" / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])

    def _memory_kb(self) -> tuple[int, int]:
        values = self._key_values(self.proc_root / "meminfo")
        return values["MemTotal"], values["MemAvailable"]

    def _process_rss_kb(self) -> int:
        return self._key_values(self.proc_root / "self" / "status").get("VmRSS", 0)

    @staticmethod
    def _key_values(path: Path) -> dict[str, int]:
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, rest = line.partition(":")
            if separator and rest.strip():
                try:
                    values[key] = int(rest.split()[0])
                except ValueError:
                    continue
        return values


class JsonlTelemetryWriter:
    def __init__(
        self,
        path: Path,
        queue_size: int = 256,
        batch_size: int = 5,
        flush_interval: float = 5.0,
        max_bytes: int = 8 * 1024 * 1024,
        backup_count: int = 2,
        run_count: int = 3,
    ) -> None:
        if max_bytes <= 0 or backup_count < 0 or run_count < 1:
            raise ValueError("Telemetry storage limits must be finite and positive")
        self.path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._run_count = run_count
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._error: str | None = None
        self._drop_lock = threading.Lock()

    @property
    def dropped(self) -> int:
        with self._drop_lock:
            return self._dropped

    @property
    def error(self) -> str | None:
        with self._drop_lock:
            return self._error

    def submit(self, record: dict) -> bool:
        with self._drop_lock:
            if self._error is not None:
                self._dropped += 1
                return False
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1
            return False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pi4-telemetry-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        output = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            runs = sorted(
                (path for path in self.path.parent.glob("*.jsonl") if path != self.path),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            retained = {self.path, *runs[:self._run_count - 1]}
            for path in self.path.parent.glob("*.jsonl*"):
                stem, separator, suffix = path.name.partition(".jsonl")
                if separator and (not suffix or (suffix.startswith(".") and suffix[1:].isdigit())):
                    run = path.with_name(stem + ".jsonl")
                    if run not in retained or (suffix and int(suffix[1:]) > self._backup_count) or path.stat().st_size > self._max_bytes:
                        path.unlink()
            pending: list[dict] = []
            last_flush = time.monotonic()
            metadata = b""
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open("rb") as previous:
                    first = previous.readline(self._max_bytes + 1)
                if json.loads(first).get("type") == "metadata":
                    metadata = first
            output = self.path.open("ab")
            size = output.tell()
            while pending or not self._stop.is_set() or not self._queue.empty():
                try:
                    pending.append(self._queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                if pending and (
                    len(pending) >= self._batch_size
                    or self._stop.is_set()
                    or time.monotonic() - last_flush >= self._flush_interval
                ):
                    for record in pending:
                        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        if record.get("type") == "metadata":
                            if len(line) >= self._max_bytes:
                                raise ValueError("Telemetry metadata exceeds file size limit")
                            metadata = line
                        elif len(metadata) + len(line) > self._max_bytes:
                            with self._drop_lock:
                                self._dropped += 1
                            continue
                        if size + len(line) > self._max_bytes:
                            output.close()
                            for index in range(self._backup_count, 0, -1):
                                source = self.path if index == 1 else self.path.with_name(f"{self.path.name}.{index - 1}")
                                if source.exists():
                                    source.replace(self.path.with_name(f"{self.path.name}.{index}"))
                            output = self.path.open("wb")
                            output.write(metadata)
                            size = len(metadata)
                            if line == metadata:
                                continue
                        output.write(line)
                        size += len(line)
                    output.flush()
                    pending.clear()
                    last_flush = time.monotonic()
        except Exception as exc:
            with self._drop_lock:
                self._error = str(exc)
        finally:
            if output is not None:
                output.close()


class TelemetryMonitor:
    def __init__(
        self,
        sampler,
        metrics: RollingMetrics,
        writer: JsonlTelemetryWriter,
        interval: float = 1.0,
        run_id: str = "",
    ) -> None:
        self._sampler = sampler
        self._metrics = metrics
        self._writer = writer
        self._interval = interval
        self._run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._latest_sample: dict | None = None
        self._sample_lock = threading.Lock()

    @property
    def samples(self) -> int:
        with self._sample_lock:
            return self._samples

    @property
    def latest_sample(self) -> dict | None:
        with self._sample_lock:
            return None if self._latest_sample is None else dict(self._latest_sample)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pi4-telemetry-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                system = self._sampler.sample()
            except Exception as exc:
                system = {"error": str(exc)}
            record = {
                "type": "sample",
                "run_id": self._run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": system,
                "runtime": self._metrics.snapshot(),
            }
            if not self._writer.submit(record):
                self._metrics.increment("telemetry_dropped")
            with self._sample_lock:
                self._latest_sample = dict(system)
                self._samples += 1
            self._stop.wait(self._interval)
