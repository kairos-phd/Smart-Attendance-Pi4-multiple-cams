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

Do not enable unattended production use until measured false-accept and false-reject results have been documented.
