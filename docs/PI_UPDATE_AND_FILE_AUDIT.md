# Updating an Existing Raspberry Pi and Per-File Audit

This guide applies to an existing Raspberry Pi 4 deployment (user `sic4b`, repository `/home/sic4b/Smart-Attendance-Pi4-multiple-cams`, data root `/var/lib/sic-attendance`, model root `/opt/sic-attendance`, such as the lab Pi at `192.168.0.208` or a hotspot address). The update workflow preserves local Pi changes—including preview/WebRTC configuration, camera paths (`icspring`, `jieli`, `sonix`), and Firebase/API settings—without blindly overwriting the checkout or losing biometric data.

## Key Runtime Fixes Included

- Missing optional media dependencies no longer prevent the HTTP API from starting. `aiortc` and `av` are declared in `pyproject.toml`, and `firebase-admin` allows `>=6.5,<8` to match packages installed on the Pi.
- Live recognition, enrollment, benchmarking, and `.npz` embedding export/import all share the same model loader and compatibility fingerprint. The fingerprint hashes the actual detector and recognizer ONNX files plus the preprocessing contract, and reads quantized output `scale` and `zero_point` directly from the ONNX graph (`QuantizeLinear`) rather than hardcoding constants. `float32`, `int8`, and `uint8` outputs are validated against an explicit output contract.
- MJPEG preview re-validates the session cookie while streaming. `PREVIEW_COOKIE_SECURE` defaults to `true`; updating the Pi requires explicitly selecting `--preview-transport http` (trusted LAN only) or `https`.
- WebRTC signaling uses Firebase RTDB transactions to claim offers and publish answers without overwriting revoked or deleted sessions. An independent watchdog closes expired peers (`WEBRTC_MAX_PEERS=3`, up to 5 minutes per peer).
- Enrollment always captures raw unannotated camera frames, while HDMI, MJPEG, and WebRTC render bounding-box overlays on a separate copy, removing a redundant per-frame copy from the capture path.
- Firestore cloud summary sync no longer initializes an unused RTDB client, and `FIREBASE_DATABASE_URL` has a safe default in `Settings`.
- `/health` publishes `model_info.files` and the active preprocessing contract so operators can verify the exact ONNX files in use.

## Staged Update from GitHub

From Windows PowerShell (or another operator host), connect to the Pi over SSH (replace `192.168.0.208` with the Pi's current IP address):

```powershell
ssh sic4b@192.168.0.208
```

Run the remaining commands **on the Pi**. Never run `git pull`, `git clean`, or `git reset --hard` on a working tree that may contain uncommitted local edits.

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams

git fetch origin main
git log -1 --oneline FETCH_HEAD

git show FETCH_HEAD:scripts/update_pi.py > /tmp/sic-update-pi.py

# Install any missing runtime dependencies without forcing upgrades on packages already within range:
.venv/bin/python -m pip install 'aiortc>=1.14,<2' 'av>=16,<18' 'firebase-admin>=6.5,<8' 'pytest>=8,<9' 'httpx>=0.27,<1' 'setuptools>=68' wheel

# Run read-only preflight checks (does not switch source or stop/restart the service):
python3 /tmp/sic-update-pi.py --repo "$PWD" --ref FETCH_HEAD
```

The script prints the backup directory `/home/sic4b/sic-backups/<timestamp>/model-preflight.json`. Inspect this JSON file to verify the resolved ONNX files, SHA-256 digests, output contract, and the count of compatible versus incompatible enrolled users. Model preflight can take several seconds while the existing service is running.

If preflight passes and you use HTTP preview on a trusted private LAN:

```bash
python3 /tmp/sic-update-pi.py --repo "$PWD" --ref FETCH_HEAD --apply --preview-transport http
sudo systemctl status sic-pi4-attendance --no-pager
curl -fsS http://127.0.0.1:8080/health | python3 -m json.tool
sudo journalctl -u sic-pi4-attendance -n 80 --no-pager
```

If preview is served over HTTPS (for example behind a TLS proxy or Cloudflare Tunnel for `sic4b.io.vn`), replace `--preview-transport http` with `--preview-transport https`. HTTP sets `PREVIEW_COOKIE_SECURE=false` (trusted LAN only); HTTPS sets `PREVIEW_COOKIE_SECURE=true`. The updater never modifies preview login credentials, `ADMIN_SHARED_TOKEN`, or camera device mappings.

During `--apply`, the script creates a local branch `pi-deploy/<timestamp>` at the verified commit. Local tracked modifications are preserved in both `git stash` and a tar archive under `/home/sic4b/sic-backups/<timestamp>`; untracked files that collide with incoming tracked paths are moved into the backup folder, while non-colliding untracked files remain in place. `.env` retains its existing values, updating only `PREVIEW_COOKIE_SECURE` and pinning `RECOGNIZER_MODEL` to the preflight-verified external recognizer path if it was previously set to `auto`. Model weights and biometric data under `/var/lib/sic-attendance` are never deleted or overwritten. After stopping the service, the user catalog is re-checked to catch any enrollment performed during preflight, and SQLite is backed up via the SQLite backup API so WAL transactions are included consistently. If service activation or `/health` verification fails, the script automatically rolls back source, `.env`, stashed changes, and moved files, leaving the backup directory intact for inspection.

> **Targeted Route Updates (No Model or Inference Changes):**
> If the Pi is running a custom or legacy recognizer whose enrolled user embeddings do not match the new full-runtime model fingerprint, do **not** force a full update. Instead, use [`scripts/update_media_only.py`](../scripts/update_media_only.py) to update only `webrtc.py` and `/api/recognition-events`, or [`scripts/update_dashboard_only.py`](../scripts/update_dashboard_only.py) (see [PI_DASHBOARD_FEED.md](PI_DASHBOARD_FEED.md)) to install `/api/dashboard` and unknown-event review routes while leaving `face_engine.py`, models, and registered user embeddings completely untouched.

## Handling Incompatible Legacy Embeddings

Earlier Pi builds recorded the model pack label (`pi4_fp32`) as `model_version` even when loading an external recognizer such as `/opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx`. The updated runtime computes a cryptographic compatibility key across the actual detector/recognizer ONNX files and preprocessing contract so embeddings from different models or quantizations are never mixed silently.

When `incompatible_users > 0`, `update_pi.py` stops **before changing source code or stopping the service**. Do not manually rewrite `users.model_version` in SQLite. Keep the database, original enrollment photos, and reference ONNX models intact, and follow [MODEL_USER_MIGRATION.md](MODEL_USER_MIGRATION.md) to verify model provenance or regenerate embeddings from consented enrollment images.

For non-production testing only, passing `--allow-incompatible` alongside `--apply` allows the update to proceed while leaving incompatible users unmatchable until migrated. It does not relabel, delete, or auto-migrate users. Never use `--allow-incompatible` for official attendance operation.

If an ONNX model fails the output contract check during preflight, the updater also aborts. Inspect the reported error and model graph/manifest before re-exporting; never guess quantization scale or zero-point values.

## Read-Only Per-File Inventory (`check_pi_files.py`)

`scripts/check_pi_files.py` is a standalone, read-only inventory tool using only the Python standard library. It classifies files across the repository, `MODEL_ROOT`, and `DATA_ROOT` without deleting anything. After updating:

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
python3 scripts/check_pi_files.py --health-url http://127.0.0.1:8080/health --output /tmp/sic-file-audit.json > /tmp/sic-file-audit.tsv
less /tmp/sic-file-audit.tsv
```

If a full update was blocked by the compatibility gate, run the audit tool directly from `FETCH_HEAD` without modifying the running service:

```bash
git show FETCH_HEAD:scripts/check_pi_files.py > /tmp/sic-check-pi-files.py
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --health-url http://127.0.0.1:8080/health --output /tmp/sic-file-audit.json > /tmp/sic-file-audit.tsv
```

If an older runtime does not expose `model_info.files` on `/health`, the audit tool conservatively marks all ONNX files in the configured pack and external recognizer path as `KEEP_ACTIVE_MODEL` and records whether the evidence came from `/health` or `.env`. Any other model file is classified as `REVIEW_MODEL`; the tool never recommends automatic deletion of model weights.

To inspect specific leftover backup or installer files observed on the Pi (works before or after updating):

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
git show FETCH_HEAD:scripts/check_pi_files.py > /tmp/sic-check-pi-files.py

for relative_file in \
  src/pi4_attendance/api.py.bak \
  src/pi4_attendance/api.py.bak2 \
  src/pi4_attendance/auth.py.bak \
  src/pi4_attendance/cloud.py.bak \
  src/pi4_attendance/face_engine.py.bak_before_qdq \
  cloudflared.deb \
  dmxcc
 do
  python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file "$PWD/$relative_file" --hash
 done

# Inspect active and backup ONNX recognizer models:
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file /opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx --hash
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file /opt/sic-attendance/models/w600k_mbf_fp32_backup.onnx --hash

# Inspect unknown binaries or installers without executing them:
file -- cloudflared.deb dmxcc
dpkg-query -W -f='${Status}\t${Version}\n' cloudflared
systemctl show sic-pi4-attendance -p FragmentPath -p DropInPaths -p WorkingDirectory -p ExecStart -p EnvironmentFiles
```

You can pass `--extra-root /etc/systemd/system --extra-root /usr/local/bin` to inventory service units and helper scripts outside the repository. `--file` reports file metadata, symlink targets, optional SHA-256 hashes, and `dpkg` package ownership without printing secret contents. Unrecognized paths default to `KEEP_REVIEW`.

| Classification | Recommended Action |
| --- | --- |
| `KEEP_PRIVATE_DATA` | Keep `.env`, credentials, SQLite databases/backups, enrollment/unknown photos, and calibration images. Never commit to GitHub. |
| `KEEP_ACTIVE_MODEL` | Keep models selected by the active runtime or `.env` configuration, including symlinks. |
| `KEEP_RUNTIME`, `KEEP_DEPENDENCY` | Keep tracked source code, deployment scripts, `.venv`, and `.git`. |
| `KEEP_VALIDATION`, `KEEP_REFERENCED` | Keep documentation, tests, and files with static references. |
| `ARCHIVE_CANDIDATE` | Unreferenced code backups (e.g. `*.bak*`): move to a separate archive directory outside the working tree after review rather than deleting blindly. |
| `REVIEW_PACKAGE` | `.deb` archives (e.g. `cloudflared.deb`): verify the package is installed and no longer needed for offline recovery before archiving elsewhere. |
| `REVIEW_MODEL` | Keep standby/backup ONNX models for A/B comparison and rollback until target-Pi benchmarks pass. |
| `REGENERABLE` | Caches and build artifacts (`__pycache__`, `.pytest_cache`, `.egg-info`); note that `.egg-info` is required for editable-install entry points. |
| `KEEP_SYSTEM`, `KEEP_REVIEW` | Keep and inspect systemd unit, package owner, or manual references before taking action. |

The JSON output is written with mode `0600`. Copy the audit reports to the operator host via `scp` if needed:

```powershell
scp sic4b@192.168.0.208:/tmp/sic-file-audit.json E:\SICHost\sic-file-audit.json
scp sic4b@192.168.0.208:/tmp/sic-file-audit.tsv E:\SICHost\sic-file-audit.tsv
```

## Optimization and Validation Roadmap

1. **Pin Verified Model Provenance:** Keep `model-preflight.json` and `/health` manifests, and pin the verified recognizer path in `.env`; never edit an old pack manifest to falsify embedding compatibility. Retain the FP32 baseline (`pi4_fp32`) and calibration images.
2. **Stable V4L2 Camera Paths:** Map all three USB cameras (`icspring`, `jieli`, `sonix`) via `/dev/v4l/by-id/...-video-index0` (or `/dev/v4l/by-path` if `by-id` is unavailable) after verifying MJPG capture at `640x480`.
3. **Bounded Multi-Camera Scheduling:** Run at `640x480`, `MONITOR_FPS=4`, testing 1, 2, and 3 cameras sequentially. Keep `FairFrameQueue` bounded (`FRAME_QUEUE_SIZE=9`), active embeddings cached in RAM, and WebRTC viewers capped (`WEBRTC_MAX_PEERS=3`). Tune FPS and cooldowns based on measured p95 latency and dropped frames rather than inflating queue capacity to hide backlog.
4. **Controlled Target-Pi Model Benchmark:** Compare the exact active detector and candidate recognizer on a consented, session-separated labeled dataset:

```bash
.venv/bin/python -m pi4_attendance.tools.benchmark \
  --model-root /opt/sic-attendance \
  --reference pi4_fp32 --candidate pi4_fp32 \
  --reference-recognizer /opt/sic-attendance/models/w600k_mbf_fp32_backup.onnx \
  --candidate-recognizer /opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx \
  --dataset /var/lib/sic-attendance/benchmark \
  --output /tmp/sic-model-benchmark.json
```

Offline dataset benchmarks report latency, embedding cosine agreement, and pairwise FAR/FRR, but must be paired with live multi-camera soak tests measuring system CPU p95, RAM/RSS, queue age/depth, temperature, throttling (`vcgencmd`), and open-set decisions on the Raspberry Pi 4 (see [DEPLOYMENT.md](DEPLOYMENT.md) and [PI_RECOG_SOAK.md](PI_RECOG_SOAK.md)).
5. **Gateway & Cloud Resilience:** Verify local SQLite continuity and idempotent outbox retry across Wi-Fi drops, SSH port-forward (`127.0.0.1:18080 -> Pi:8080`) restarts, FastAPI gateway (`127.0.0.1:8081`) restarts, and Cloudflare Tunnel (`sic4b.io.vn`) reconnects.
