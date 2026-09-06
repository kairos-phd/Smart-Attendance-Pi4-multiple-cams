#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/scripts/install.sh"
bash "$ROOT/scripts/verify_hardware.sh"
bash "$ROOT/scripts/install_systemd.sh"

cat <<EOF

Bootstrap complete. Continue in this order:
  1. Edit $ROOT/.env and set camera paths plus ADMIN_SHARED_TOKEN.
  2. Prepare FP32 models:
     $ROOT/.venv/bin/sic-pi4-prepare-models --model-root /opt/sic-attendance --accept-insightface-noncommercial-license
  3. Verify locally:
     cd $ROOT && .venv/bin/pytest -q && .venv/bin/sic-pi4
  4. Start now and on boot, then reboot once to verify:
     sudo systemctl enable --now sic-pi4-attendance
     sudo reboot
EOF
