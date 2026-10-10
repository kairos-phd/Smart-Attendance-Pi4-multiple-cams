# Recognition soak: 30 minutes on the existing Pi runtime

This collector uses only Python's standard library and works alongside the running service. It does not install dependencies, checkout source, change models, enroll users, or restart the service. Keep the Pi's current legacy recognizer for this run. Recognition must see real camera traffic: place known and unknown faces in view, vary distance/light, and include periods with multiple faces and no face. This measures the workload you present; it does not generate camera frames or synthetic CPU load.

Run these commands in Windows PowerShell. SSH connects to the Pi; the HTTP API is accessed only through loopback **on the Pi**, not from Windows.

```powershell
# Fetch main and extract ONLY the standalone collector into /tmp.
ssh sic4b@192.168.0.208 'cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams && git fetch origin main && git show FETCH_HEAD:scripts/stress_recognition.py > /tmp/stress_recognition.py'

# Optional 10-second smoke run first; inspect its report before the full run.
ssh sic4b@192.168.0.208 'python3 /tmp/stress_recognition.py --duration 10 --interval 1 --data-root /var/lib/sic-attendance --output-dir /tmp/sic-recog-smoke --enable-recognition'
ssh sic4b@192.168.0.208 'cat /tmp/sic-recog-smoke/report.txt'

# Keep this SSH session open for the full 30-minute run.
ssh sic4b@192.168.0.208 'python3 /tmp/stress_recognition.py --duration 1800 --interval 5 --url http://127.0.0.1:8080 --data-root /var/lib/sic-attendance --output-dir /tmp/sic-recog-30m --enable-recognition --journal'

# Archive after the collector exits, including if interrupted with Ctrl+C.
ssh sic4b@192.168.0.208 'cat /tmp/sic-recog-30m/report.txt; tar -C /tmp -czf /tmp/sic-recog-30m.tar.gz sic-recog-30m'
scp sic4b@192.168.0.208:/tmp/sic-recog-30m.tar.gz 'F:\sic-recog-30m.tar.gz'
```

Output directories must be new: choose a different name on reruns instead of deleting an earlier report. The default duration is 1800 seconds, interval 5 seconds, data root `/var/lib/sic-attendance`, and API origin `http://127.0.0.1:8080`. A unique output directory under `/tmp` is selected when `--output-dir` is omitted. `--duration` and `--interval` accept fractional seconds down to 0.1 for smoke tests.

`--enable-recognition` explicitly authorizes changing only the local recognition flag. The collector reads the prior flag, enables recognition if necessary, verifies it, and restores the prior flag in `finally` after normal completion, startup errors, or Ctrl+C. Without this option it only observes. A restore failure is recorded prominently in the report and exits with code 1; verify `/api/control` locally before leaving the Pi unattended. SIGKILL, loss of power, or a permanently unreachable API cannot guarantee restoration. Keep the SSH connection open; do not stop the run by killing the service.

The archive contains:

- `health.jsonl`: timestamped health/control samples, Pi temperature and throttle bitmask when available. CPU and RAM come from service health telemetry, so process CPU/RSS describe the attendance service rather than the collector.
- `events.jsonl`: only newly appended SQLite recognition events after the initial maximum rowid. Database access is read-only; event exports omit names and user/student IDs. The old runtime may have no `recognition_events` table; that is reported as unavailable rather than replaying older attendance history or treating an empty log as success.
- `summary.json` and `report.txt`: actual duration, recognition-enabled sample count, CPU/queue p95, RAM/RSS p95, peak temperature, event decision-delay p95, available rolling latency p95 peaks, dropped-counter deltas, model versions, and warnings. Missing metrics remain `null`.
- `service.json`: service state/PID/restart counter before and after the run.
- `service-journal.txt`: optional, up to the latest 10,000 journal lines in the run interval, readable by the current account. No sudo or credentials are collected; lack of journal access becomes a warning. Journal text may include operational identifiers; inspect it before sharing.

Sample p95 uses linear interpolation over sorted values. System CPU is aggregate utilization across all logical cores (0–100%). The repository's health telemetry normalizes service process CPU by logical CPU count (also 0–100%); these are separate metrics. Queue depth is a gauge. `queue_age_ms`, `inference_ms`, and `full_processing_ms` come from runtime rolling distributions when present: the summary reports their **largest observed rolling p95**, not a whole-run p95 reconstructed from percentiles. Event decision delay is the exact difference between timezone-aware `decided_at` and `observed_at`, including decision/voting delays. It is not pure model inference time. Temperature/throttle availability depends on the Pi tools/files; throttle masks include both active and historical flags. Counter resets make their deltas unavailable.

A collector exit code 0 means collection completed without API or restore errors; inspect recognition-enabled counts, inference health, events, and warnings to judge the run. It is not a hardware or recognition-quality pass. Without labeled ground truth, this report cannot measure FAR, FRR, or accuracy. There is no claimed real-Pi 30-minute result until these commands finish on the Pi.
