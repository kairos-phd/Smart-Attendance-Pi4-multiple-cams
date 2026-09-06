# SIC Attendance for Raspberry Pi 4

This folder is the self-contained Raspberry Pi 4 deployment for realtime biometric attendance. The Pi performs capture, optional motion triggering, face detection, recognition, enrollment, decision metadata, and biometric storage. A local PC is not part of the runtime path.

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

## Quick deployment

```bash
git clone https://github.com/kairos-phd/Smart-Attendance-Pi4-multiple-cams.git sic-attendance
cd sic-attendance
bash scripts/bootstrap_pi.sh
nano .env
```

Place at least 20 representative, consented calibration images under the microSD data directory, then generate both candidates:

```bash
.venv/bin/sic-pi4-prepare-models \
  --model-root /opt/sic-attendance \
  --calibration-dir /var/lib/sic-attendance/calibration \
  --accept-insightface-noncommercial-license
```

Benchmark on the Pi:

```bash
.venv/bin/sic-pi4-benchmark \
  --model-root /opt/sic-attendance \
  --dataset /var/lib/sic-attendance/calibration
```

Keep `MODEL_PACK=pi4_fp32` unless the INT8 benchmark and recognition evaluation pass. Then test manually:

```bash
.venv/bin/pytest -q
.venv/bin/sic-pi4
```

## HDMI preview

Close Guvcview and stop the background service before opening the integrated preview. Run this from a Terminal in the Pi desktop session, without `sudo`:

```bash
sudo apt install -y python3-tk
.venv/bin/python -m pip install -e '.[test]'
sudo systemctl stop sic-pi4-attendance
.venv/bin/sic-pi4 --preview
```

The selected camera previews at up to `PREVIEW_FPS=30`; recognition remains bounded by `MONITOR_FPS=4`, motion detection, and the camera trigger cooldown. Recognition starts off. Press `A` to toggle recognition, `1`/`2`/`3` to select a camera, and `Q` or `Esc` to exit. Restart unattended operation afterward with `sudo systemctl start sic-pi4-attendance`.

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
