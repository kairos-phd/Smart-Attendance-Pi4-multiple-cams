# Raspberry Pi 4 deployment

## Hardware

- Raspberry Pi 4 Model B, 4 GB or more.
- 64-bit Raspberry Pi OS based on Debian 13 (Trixie).
- Official-quality 5 V / 3 A supply.
- Active fan or well-tested cooling.
- One or more USB webcams or a separately validated mixed/multiplexer arrangement.
- Externally powered USB hub when camera power or stability requires it.
- High-endurance A2 microSD, 64 GB or larger.

The Pi 4's USB 2 devices share a hub and bandwidth. Configure cameras at 640×480 and 4 monitor FPS initially; identify stable `/dev/v4l/by-id` paths and test every configured camera simultaneously.

## Configure cameras

List every V4L2 device and stable camera path:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id
```

For each camera, select its `/dev/v4l/by-id/...-video-index0` symlink only after confirming that it exposes the required capture format:

```bash
CAM=/dev/v4l/by-id/REPLACE_CAMERA-video-index0
v4l2-ctl --device="$CAM" --list-formats-ext
v4l2-ctl --device="$CAM" --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30 --stream-mmap=3 --stream-count=120 --stream-to=/dev/null
```

Run the two commands again with each camera symlink. If `/dev/v4l/by-id` is unavailable, use `/dev/v4l/by-path` and keep the physical USB ports unchanged.

Set `.env` to the number of connected cameras. Keep unused `CAMERA_n_*` entries in the file; the runtime reads only entries from `1` through `CAMERA_COUNT`.

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

Every `CAMERA_n_ID` must be unique. Stop Guvcview before testing or starting SIC; Guvcview and the runtime must not open the same camera concurrently.

## Installation

Clone the repository root and run `bash scripts/bootstrap_pi.sh`. It validates Debian 13/aarch64, installs the Trixie `t64` runtime package, creates the virtual environment and runtime directories, checks attached hardware, and installs/enables the systemd unit. It preserves an existing `.env`.

Edit `.env` with exact `/dev/v4l/by-id` camera paths and a generated administrator token. Store Firebase credentials under `/opt/sic-attendance/secrets` with mode `0600`. `MODEL_ROOT=/opt/sic-attendance` means InsightFace packs live at `/opt/sic-attendance/models/<pack>`.

Prepare and benchmark the models before starting the service. Test each camera alone and all configured cameras together. Confirm that camera health, queue age/depth, inference latency, dropped frames, thermals, and throttling remain acceptable.

```bash
.venv/bin/sic-pi4-prepare-models --model-root /opt/sic-attendance --accept-insightface-noncommercial-license
.venv/bin/pytest -q
.venv/bin/sic-pi4
sudo systemctl enable --now sic-pi4-attendance
sudo systemctl status sic-pi4-attendance
curl http://127.0.0.1:8080/health
sudo reboot
```

## Required tests

1. Unit tests pass.
2. Every configured camera operates for at least 4–8 hours.
3. Disconnect/reconnect every camera.
4. Remove network access; local attendance must continue.
5. Interrupt power during idle and test database recovery.
6. Enroll at least two consented test users using 3–8 views.
7. Test genuine, impostor, poor-light, angled, glasses, and printed-photo cases.
8. Verify one accepted observation creates attendance and cooldown suppresses repeats.
9. Verify weak similarity, margin, detection confidence, or quality creates a rejected event.
10. Verify unknown retention stores only the matching face crop and removes expired files.

## Service operations

```bash
sudo systemctl start sic-pi4-attendance
sudo systemctl status sic-pi4-attendance
journalctl -u sic-pi4-attendance -f
curl http://127.0.0.1:8080/health
```

## HDMI preview

Guvcview and the attendance runtime must not open the same camera concurrently. Close Guvcview, then run the integrated preview from a Terminal in the active Pi desktop session:

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
sudo apt install -y python3-tk
.venv/bin/python -m pip install -e '.[test]'
sudo systemctl stop sic-pi4-attendance
.venv/bin/sic-pi4 --preview
```

The selected camera requests MJPG at `PREVIEW_FPS=30`; other cameras remain at `MONITOR_FPS`. Press `A` to toggle recognition, `1`/`2`/`3` to switch camera, and `Q` or `Esc` to stop.

To enroll from the preview, press `E` and enter the name and student ID. Press `C` to capture each new frame, using 3–8 clear views with exactly one face per frame. Press `Enter` to save or `X` to cancel. The overlay reports progress and errors; a successful enrollment enters the active recognition catalog without a restart.

## Compare one, two, and three cameras

`--preview` automatically creates one JSONL performance log per run in `/var/lib/sic-attendance/performance`. Monitoring samples once per second and writes through a bounded background queue, so telemetry cannot wait on the inference thread.

For each camera count, set `CAMERA_COUNT` in `.env`, keep the same selected preview camera, start the preview, press `A`, reproduce the same motion/face workload for 10–15 minutes, then press `Q`. Warm up for two minutes and repeat each configuration three times.

```bash
sed -i 's/^CAMERA_COUNT=.*/CAMERA_COUNT=1/' .env
.venv/bin/sic-pi4 --preview

sed -i 's/^CAMERA_COUNT=.*/CAMERA_COUNT=2/' .env
.venv/bin/sic-pi4 --preview

sed -i 's/^CAMERA_COUNT=.*/CAMERA_COUNT=3/' .env
.venv/bin/sic-pi4 --preview
```

Only select a count after all matching `CAMERA_n_SOURCE` paths have been configured and tested. Inspect the newest sample and preserve each complete log for comparison:

```bash
LOG=$(ls -1t /var/lib/sic-attendance/performance/*.jsonl | head -n 1)
head -n 1 "$LOG" | python3 -m json.tool
tail -n 1 "$LOG" | python3 -m json.tool
vcgencmd measure_temp
vcgencmd get_throttled
```

Compare system CPU/RAM, process RSS, `camera.<id>.captured` rate, `camera.<id>.enqueued` rate, `inference.processed` rate, `queue.depth`, queue replacements, and p50/p95/p99 queue/inference/full-processing latency. Investigate FP32 as too heavy when queue pressure persists, processed throughput falls below admitted work, CPU stays above about 85%, available RAM falls below about 512 MB, swap or throttling appears, or full-processing latency exceeds the attendance requirement. First separate inference pressure from camera read failures, USB bandwidth, preview load, storage, and thermal throttling; INT8 only addresses an inference-bound result.

Run `sudo systemctl start sic-pi4-attendance` afterward to restore unattended operation.

Do not enable unattended production use until measured false-accept and false-reject results have been documented.
