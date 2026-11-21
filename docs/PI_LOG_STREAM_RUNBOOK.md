# Console Log, Recognition Toggle, and Pi Streaming Runbook

## Log Verification Summary (2026-10-10 Baseline)

- Commit `52a5436` passed **367 tests on the Pi**, and the ONNX preflight reader verified the external quantized recognizer output contract with INT8 scale `0.02524387650191784` and zero point `0`.
- Full runtime update paused at the compatibility gate because **existing active users did not match the new model fingerprint** (subsequently expanded to 3 registered users in the `2026-10-24` project report), keeping the existing running service intact. The absence of `model_info` on `/health` was expected on the pre-update runtime rather than metadata loss after an update.
- The running service had recorded `recognition.rejected=1764`; because the legacy `/api/recognitions` route filtered for `accepted` events only, the console log could appear empty while inference was actively processing frames. This was a feed-filtering limitation rather than SQLite data loss.
- At the `/health` snapshot with `recognition_enabled=0`, all three USB cameras (`icspring`, `jieli`, `sonix`) were capturing at approximately `4 FPS` (`640x480`). Resource readings taken while recognition is disabled must not be used as active inference benchmarks, and WebRTC media delivery requires verifying browser inbound frames/packets rather than signaling state alone.

## Inspecting Recognition Logs on an Existing Pi Runtime

The commands below fetch only the read-only watcher script and inspect the existing SQLite database and loopback control state. They do not modify the runtime, stop the service, or alter model fingerprints or stored embeddings:

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
git fetch origin main
git show FETCH_HEAD:scripts/watch_recognition.py > /tmp/sic-watch-recognition.py

# View the 20 most recent recognition events and current runtime state:
python3 /tmp/sic-watch-recognition.py --repo "$PWD" --once

# Run continuously in a separate terminal while toggling Recognition in the console (Ctrl+C to stop):
umask 077
python3 -u /tmp/sic-watch-recognition.py --repo "$PWD" --limit 20 \
  | tee "/tmp/sic-recognition-$(date +%Y%m%d-%H%M%S).jsonl"
```

The JSONL output uses the SQLite recognition event ID and records the camera ID, timestamps, similarity, decision reason, and model version. `recognition_state` logs `enabled`, current model, previous model, and `model_changed`. Toggling recognition only flips the admission processing flag; it does not reload or switch models. Polling may miss a rapid toggle that starts and finishes between two poll intervals, whereas the systemd journal on updated runtimes records every API toggle.

An `unknown` display status means a face observation was evaluated but had `no_candidate` or `similarity_below_threshold`; low face quality, low detection score, or insufficient top-1/top-2 margin produces `rejected` with an explicit reason. Frames without a detected face do not generate synthetic unknown or attendance events. Unknown observations are never counted as attendance, and rejected observations never expose a candidate user's name as an accepted match.

If the database is restored or truncated and `rowid` moves backward, the watcher emits `log_cursor_reset` and may replay older events; deduplicate by `observation_id` when aggregating JSONL files. Replaying watcher output never creates new attendance rows.

The watcher opens SQLite in read-only mode (`mode=ro`), performs no schema migration, and uploads nothing to the cloud. The Linux user must have read permissions on `DATA_ROOT`, the SQLite database, and its WAL files. Only `DATA_ROOT` and `API_PORT` are read from `.env`; tokens and credentials are never printed. If the systemd service uses an `EnvironmentFile` outside the repository `.env`, pass `--data-root` and `--url http://127.0.0.1:PORT` to match the running service. Keep exported logs private because they contain attendance timestamps and identifiers.

If `diagnostic_error` appears in the stream, preserve the full line for debugging. `HTTP 401`/`403` responses on protected API endpoints indicate an authentication mode or token mismatch; never make administrative endpoints public to bypass authentication errors. The watcher uses the loopback control endpoint alongside direct read-only SQLite access to diagnose older runtimes safely.

## Verifying Streaming and Inference When Toggling Recognition

Open three SSH terminals on the Pi. In Terminal 1, monitor the service journal:

```bash
sudo journalctl -u sic-pi4-attendance --since '10 minutes ago' \
  --no-pager -l -o short-iso \
  | rg -i 'recognition|inference|webrtc|ice|traceback|error|failed'

sudo journalctl -u sic-pi4-attendance -f -l -o short-iso
```

In Terminal 2, run `watch_recognition.py` as shown above. In Terminal 3, capture health and control snapshots before and after enabling recognition:

```bash
umask 077
curl -fsS http://127.0.0.1:8080/health > /tmp/sic-health-recog-off.json
curl -fsS http://127.0.0.1:8080/api/control | python3 -m json.tool

# Enable Recognition in the console or web dashboard, have consented enrolled and unknown subjects
# pass in front of each camera, wait for the sampling window, then capture the active state:
curl -fsS http://127.0.0.1:8080/health > /tmp/sic-health-recog-on.json
curl -fsS http://127.0.0.1:8080/api/control | python3 -m json.tool
vcgencmd measure_temp
vcgencmd get_throttled
```

Compare `captured`, `enqueued`, and `processed` frame rates, `recognition_enabled`, inference/queue p95 latencies, and CPU utilization. `camera_health.last_seen` alone does not replace checking that captured frame counters advance. Video preview frames are served from the capture worker independent of whether a frame is admitted to AI inference: when recognition is off, live video still streams; when recognition is on, preview frames display bounding-box overlays while enrollment continues to use raw unannotated frames.

Automated tests verify raw-to-overlay-to-raw frame transitions and monotonic RTP timestamps across all three cameras (`icspring`, `jieli`, `sonix`) when toggling recognition. On physical hardware, also verify in the browser that video frames advance and WebRTC `getStats()` shows increasing inbound frames and packets; an established ICE state without advancing video frames is not a passing stream test. STUN alone may not traverse symmetric NAT or hotspot isolation, where an authenticated TURN relay is required.

## Updated Runtime Endpoints and Web Gateway Topology

- `sic-console` defaults to `/api/recognition-events`; `--accepted-only` retains the legacy accepted-only view when needed.
- `/api/recognitions` preserves the legacy accepted-only contract, and `/api/attendance` returns deduplicated attendance records outside `ATTENDANCE_COOLDOWN_SECONDS`.
- `/api/recognition-events` returns `observation_id`, `display_status` (`accepted`, `unknown`, `rejected`), `status`, `reason`, `model_version`, camera ID, timestamps, and scores.
- `/api/control/remote` (`GET`/`PATCH`) requires administrator bearer authentication, whereas `/api/control` remains restricted to loopback (`127.0.0.1`/`::1`). Patching `recognition_enabled` applies globally across all configured cameras; `camera_id` selects the active preview camera.
- `/api/dashboard` and `PATCH /api/unknown-events/{observation_id}/review` (see [PI_DASHBOARD_FEED.md](PI_DASHBOARD_FEED.md)) expose read-only SQLite users, grouped attendance, camera health, and durable unknown-event review metadata (`dashboard_reviews.sqlite3`) to the web gateway (`VITE_DATA_SOURCE=pi`).
- The Pi administrator token (`ADMIN_SHARED_TOKEN`) is stored only in server-side configuration (`.env` on the Pi and `.env.media` on the Windows gateway host at `E:\SICHost`), never embedded in frontend `VITE_*` variables, client JavaScript, or URLs.

Because full runtime updates enforce strict model fingerprint matching, **do not apply a full inference overwrite onto a Pi whose enrolled user catalog was created with an older model fingerprint** until original enrollment images are verified or embeddings are regenerated under [MODEL_USER_MIGRATION.md](MODEL_USER_MIGRATION.md). Never rewrite `model_version` in SQL or use `--allow-incompatible` for production attendance. To add WebRTC, recognition-event logs, or dashboard feeds to an existing Pi without touching its model or user catalog, use the targeted updaters ([`scripts/update_media_only.py`](../scripts/update_media_only.py) and [`scripts/update_dashboard_only.py`](../scripts/update_dashboard_only.py)) or follow [PI_UPDATE_AND_FILE_AUDIT.md](PI_UPDATE_AND_FILE_AUDIT.md).

## Next Verification Steps

1. Preserve watcher JSONL logs, `/health` snapshots with recognition off/on, systemd journal excerpts, and `model-preflight.json`. Audit enrolled users and original enrollment images before any full runtime or model migration.
2. Benchmark 1, 2, and 3 cameras (`icspring`, `jieli`, `sonix`) under controlled conditions with enrolled and unknown holdout subjects, recording system CPU p95, inference/queue/full-processing p95 latency, temperature/throttling, and labeled FAR/FRR. Test 1–3 concurrent WebRTC viewers to separate inference load from video encoding overhead.
3. Verify offline operation and recovery by disconnecting Wi-Fi/cloud, shutting down the host gateway (`127.0.0.1:8081` / SSH forward `127.0.0.1:18080 -> Pi:8080` / Cloudflare Tunnel `sic4b.io.vn`), and confirming local SQLite persistence and idempotent outbox retry on reconnection.
4. Expand the consented, session-separated evaluation dataset and evaluate presentation attack detection (PAD) before unattended deployment.
