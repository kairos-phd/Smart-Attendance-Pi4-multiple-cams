# Raspberry Pi 4 deployment

## Hardware

- Raspberry Pi 4 Model B, 4 GB or more.
- 64-bit Raspberry Pi OS Bookworm or newer.
- Official-quality 5 V / 3 A supply.
- Active fan or well-tested cooling.
- Three USB webcams or a separately validated mixed/multiplexer arrangement.
- Externally powered USB hub when camera power or stability requires it.
- High-endurance A2 microSD, 64 GB or larger.

The Pi 4's USB 2 devices share a hub and bandwidth. Configure cameras at 640×480 and 4 monitor FPS initially; identify stable `/dev/v4l/by-id` paths and test all three simultaneously.

## Installation

Run `scripts/verify_hardware.sh`, then `scripts/install.sh`. Edit `.env` with exact device paths. Store Firebase credentials under `/opt/sic-attendance/secrets` with mode `0600`.

Prepare and benchmark the models before starting the service. Test each camera alone, each pair, and all three. Confirm that camera health, queue depth, thermals, and throttling remain acceptable.

## Required tests

1. Unit tests pass.
2. Three cameras operate for at least 4–8 hours.
3. Disconnect/reconnect every camera.
4. Remove network access; local attendance must continue.
5. Interrupt power during idle and test database recovery.
6. Enroll at least two consented test users using 3–8 views.
7. Test genuine, impostor, poor-light, angled, glasses, and printed-photo cases.
8. Verify no single camera can create attendance.
9. Verify conflicting identities create a rejected conflict.
10. Verify unknown-image retention removes expired files.

## Service operations

```bash
sudo systemctl start sic-pi4-attendance
sudo systemctl status sic-pi4-attendance
journalctl -u sic-pi4-attendance -f
curl http://127.0.0.1:8080/health
```

Do not enable unattended production use until measured false-accept and false-reject results have been documented.

