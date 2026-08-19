# SIC Attendance for Raspberry Pi 4

This is the self-contained Raspberry Pi 4 deployment for three-camera, vote-confirmed attendance. The Pi performs capture, motion triggering, face detection, recognition, enrollment, voting, and biometric storage. A local PC is not part of the runtime path.

## Decision rules

- Exactly three configured cameras.
- Attendance requires matching observations from at least two distinct cameras inside a configurable time window.
- Repeated frames from one camera count as one vote.
- Weak, ambiguous, or conflicting identities do not create attendance.
- When fewer than two cameras are available, automatic confirmation must not be trusted.
- Images remain `uint8` JPEG data; face embeddings are symmetric per-vector INT8 BLOBs.
- Biometric images, embeddings, and datasets remain on the Pi microSD.
- Model INT8 conversion is accepted only after a target-Pi benchmark and identity accuracy evaluation.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/pi4_attendance` | Runtime, camera queue, inference, voting, SQLite, and protected API |
| `src/pi4_attendance/tools` | Model preparation and target-Pi benchmark tools |
| `models` | Licensing and reproducible model-generation instructions; no restricted weights |
| `scripts` | Installation, hardware verification, and systemd installation |
| `systemd` | Hardened boot service |
| `tests` | INT8, voting, and local-storage tests |
| `docs` | Architecture, deployment, calibration, storage, security, and acceptance guidance |

## Quick deployment

```bash
git clone <YOUR_REPOSITORY_URL> sic-attendance
cd sic-attendance/PI4
bash scripts/verify_hardware.sh
bash scripts/install.sh
nano .env
```

Place at least 20 representative, consented calibration images under the microSD data directory, then generate both candidates:

```bash
.venv/bin/sic-pi4-prepare-models \
  --model-root /opt/sic-attendance/models \
  --calibration-dir /var/lib/sic-attendance/calibration \
  --accept-insightface-noncommercial-license
```

Benchmark on the Pi:

```bash
.venv/bin/sic-pi4-benchmark \
  --model-root /opt/sic-attendance/models \
  --dataset /var/lib/sic-attendance/calibration
```

Keep `MODEL_PACK=pi4_fp32` unless the INT8 benchmark and recognition evaluation pass. Then test manually:

```bash
.venv/bin/pytest -q
.venv/bin/sic-pi4
```

Install at boot only after all three cameras and enrollment have been verified:

```bash
bash scripts/install_systemd.sh
sudo reboot
sudo systemctl status sic-pi4-attendance
journalctl -u sic-pi4-attendance -f
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

Read [deployment](docs/DEPLOYMENT.md), [voting](docs/VOTING.md), [model optimization](docs/MODEL_OPTIMIZATION.md), [data storage](docs/DATA_STORAGE.md), and [security](docs/SECURITY.md) before installation.
