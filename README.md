# SIC Attendance for Raspberry Pi 4

This folder is the self-contained Raspberry Pi 4 deployment for realtime biometric attendance. The Pi performs capture, optional motion triggering, face detection, recognition, enrollment, decision metadata, and biometric storage. A local PC is not part of the runtime path.

For an existing Pi with local source changes, follow [Cập nhật Pi và kiểm kê từng file](docs/PI_UPDATE_AND_FILE_AUDIT.md). It preserves local work, checks the actual model and embedding compatibility, and provides a read-only file inventory before cleanup.

## Decision rules

- One or more uniquely identified cameras may be configured.
- Each detected face produces an independent biometric observation.
- Attendance requires similarity, margin, detection-confidence, and face-quality thresholds to pass.
- Repeated accepted observations are suppressed by the global per-user attendance cooldown.
- Camera ID is provenance and health metadata, not a quorum requirement.
- Images remain `uint8` JPEG data; face embeddings are symmetric per-vector INT8 BLOBs.
- Biometric images, embeddings, and datasets remain on the Pi microSD.
- Model INT8 conversion is accepted only after a target-Pi benchmark and identity accuracy evaluation.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/pi4_attendance` | Runtime, camera queue, inference, recognition policy, SQLite, and protected API |
| `src/pi4_attendance/tools` | Model preparation and target-Pi benchmark tools |
| `models` | Licensing and reproducible model-generation instructions; no restricted weights |
| `scripts` | Installation, hardware verification, and systemd installation |
| `systemd` | Hardened boot service |
| `tests` | Recognition, image-boundary, model, storage, API, and migration tests |
| `docs` | Architecture, deployment, calibration, storage, security, and acceptance guidance |

## Add cameras

On the Pi, list the V4L2 devices and their stable symlinks:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id
```

Use each camera's `/dev/v4l/by-id/...-video-index0` symlink when it supports video capture. Confirm it before editing `.env`:

```bash
CAM=/dev/v4l/by-id/REPLACE_CAMERA-video-index0
v4l2-ctl --device="$CAM" --list-formats-ext
v4l2-ctl --device="$CAM" --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30 --stream-mmap=3 --stream-count=120 --stream-to=/dev/null
```

Set `CAMERA_COUNT=1`, `2`, or `3` only when `.env` has a matching stable `/dev/v4l/by-id` entry for every enabled camera. Keep `CAMERA_COUNT=1` while only camera 1 is connected. To add camera 2 or 3, set `CAMERA_COUNT` to `2` or `3` and fill the matching entries:

```dotenv
CAMERA_COUNT=3
CAMERA_1_ID=camera-1
CAMERA_1_NAME=Camera 1
CAMERA_1_SOURCE=/dev/v4l/by-id/REPLACE_CAMERA_1-video-index0
CAMERA_2_ID=camera-2
CAMERA_2_NAME=Camera 2
CAMERA_2_SOURCE=/dev/v4l/by-id/REPLACE_CAMERA_2-video-index0
CAMERA_3_ID=camera-3
CAMERA_3_NAME=Camera 3
CAMERA_3_SOURCE=/dev/v4l/by-id/REPLACE_CAMERA_3-video-index0
```

Give every camera a unique `CAMERA_n_ID`. If `/dev/v4l/by-id` is unavailable, use `/dev/v4l/by-path` and keep each camera on the same USB port. Close Guvcview before testing or starting SIC because only one process can own a camera reliably.

## Quick deployment

```bash
git clone https://github.com/kairos-phd/Smart-Attendance-Pi4-multiple-cams.git sic-attendance
cd sic-attendance
bash scripts/bootstrap_pi.sh
nano .env
TOKEN=$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')
sed -i "s|^ADMIN_SHARED_TOKEN=.*|ADMIN_SHARED_TOKEN=$TOKEN|" .env
chmod 600 .env
```

Place at least 20 representative, consented calibration images under the microSD data directory, then generate both candidates:

```bash
.venv/bin/sic-pi4-prepare-models \
  --model-root /opt/sic-attendance \
  --calibration-dir /var/lib/sic-attendance/calibration \
  --accept-insightface-noncommercial-license
```

Keep `MODEL_PACK=pi4_fp32` unless the INT8 benchmark and recognition evaluation pass. Then test manually:

```bash
.venv/bin/pytest -q
.venv/bin/sic-pi4
```

## Production runtime and headless console

One `sic-pi4` runtime owns all configured 1–3 cameras, the shared `FairFrameQueue`, one InsightFace worker, recognition policy, SQLite, cloud outbox, and Firebase sync. All configured cameras recognize simultaneously. The console panes are loopback clients only; they never start another runtime or own a camera.

For headless operation, start the service, then attach the console:

```bash
sudo systemctl start sic-pi4-attendance
sic-console
```

The tmux layout is left `sic-pi4-control`, top-right `sic-pi4-performance`, and bottom-right `sic-pi4-attendance-log`. The lower-right pane shows accepted recognitions in the Pi's local time, including repeated recognitions suppressed by the attendance cooldown, refreshing every second by default. Click the control pane and check its `ACTIVE` marker before using keys. The control pane keys are `A` recognition toggle, `E` enrollment start, `1`/`2`/`3` camera selection, `C` capture, `Enter` submit, `X` cancel, and `Q` close console client. `Q` closes only the console client, not `sic-pi4-attendance`.

Headless enrollment reuses the selected camera's latest `CameraWorker` frame once per sequence and never reopens a camera. The attendance viewer labels cosine similarity as `Similarity`, not a confidence percentage.

Headless and preview runtimes write 1 Hz JSONL telemetry through a bounded nonblocking writer under `DATA_ROOT/performance`. Logs rotate at 8 MiB with two backups per run and retain the three newest runs (at most 72 MiB); rotation keeps model/camera metadata and removes expired diagnostic logs. Inspect a running service with:

```bash
sic-pi4-performance --once
LOG=$(ls -1t /var/lib/sic-attendance/performance/*.jsonl | head -n 1)
tail -n 1 "$LOG" | python3 -m json.tool
curl http://127.0.0.1:8080/health | python3 -m json.tool
```

## Target-Pi benchmark

The benchmark reads dataset images and does not own cameras. Stop the service to avoid CPU/RAM contention and distorted measurements, then restart it afterward:

```bash
sudo systemctl stop sic-pi4-attendance
.venv/bin/sic-pi4-benchmark --model-root /opt/sic-attendance --dataset /var/lib/sic-attendance/calibration
sudo systemctl start sic-pi4-attendance
```

## HDMI preview

Preview is optional: it shows one selected camera while every configured camera continues recognizing in the background. Close Guvcview and stop the background service before opening it; preview cannot run beside `sic-pi4-attendance` because both would own cameras. Run this from a Terminal in the Pi desktop session, without `sudo`:

```bash
sudo apt install -y python3-tk
.venv/bin/python -m pip install -e '.[test]'
sudo systemctl stop sic-pi4-attendance
.venv/bin/sic-pi4 --preview
```

The selected camera previews at up to `PREVIEW_FPS=30`; recognition remains bounded by `MONITOR_FPS=4`, motion detection, and the camera trigger cooldown. Recognition starts off. Press `A` to toggle recognition, `1`/`2`/`3` to select a camera, and `Q` or `Esc` to exit.

Press `E` to enroll from the preview, enter the name and student ID, then press `C` for each new view. Capture 3–8 clear views with one face in each frame. Press `Enter` to save or `X` to cancel. Enrollment runs in the background and the overlay shows its result. A successful enrollment is available to recognition immediately.

Each headless or preview run writes a nonblocking 1 Hz performance log under `DATA_ROOT/performance`. The first row describes the model, camera configuration, and requested resolution; later JSONL rows contain system CPU/RAM, process CPU/RSS, measured camera capture/enqueue rates, inference rate, queue pressure, and rolling latency percentiles. View the current log while recognition is running:

```bash
LOG=$(ls -1t /var/lib/sic-attendance/performance/*.jsonl | head -n 1)
tail -n 1 "$LOG" | python3 -m json.tool
curl http://127.0.0.1:8080/health | python3 -m json.tool
```

The attendance API is the deduplicated recognition log. It returns the name, student ID, camera IDs, timestamp, and cosine similarity; repeated accepted frames inside `ATTENDANCE_COOLDOWN_SECONDS` remain diagnostic recognition events and do not create another attendance row.

```bash
curl -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8080/api/attendance?limit=20" | python3 -m json.tool
```

Restart unattended operation afterward with `sudo systemctl start sic-pi4-attendance`.

Windows tests do not prove real Raspberry Pi 2/3-camera operation. Validate real cameras, latency, RAM, thermals, queue pressure, and recovery on the target Pi before production use.

Install at boot only after every configured camera and enrollment have been verified:

```bash
sudo systemctl enable --now sic-pi4-attendance
sudo systemctl status sic-pi4-attendance
journalctl -u sic-pi4-attendance -f
sudo reboot
sudo systemctl status sic-pi4-attendance
```

## Enrollment API

The API listens on port `8080` by default. `/health` is public; biometric administration routes require an administrator bearer token.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -F "name=Example User" \
  -F "student_id=S001" \
  -F "face_images=@front.jpg" \
  -F "face_images=@left.jpg" \
  -F "face_images=@right.jpg" \
  http://PI_HOST:8080/api/users
```

Real face images, `.env`, service credentials, generated model packs, SQLite databases, and benchmark results are ignored by Git.

Read [deployment](docs/DEPLOYMENT.md), [recognition policy](docs/VOTING.md), [model optimization](docs/MODEL_OPTIMIZATION.md), [data storage](docs/DATA_STORAGE.md), and [security](docs/SECURITY.md) before installation.
