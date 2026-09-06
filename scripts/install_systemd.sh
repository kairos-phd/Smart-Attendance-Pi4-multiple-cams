#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-$USER}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
SERVICE_PATH="/etc/systemd/system/sic-pi4-attendance.service"

sed -e "s|REPLACE_USER|$SERVICE_USER|g" -e "s|REPLACE_GROUP|$SERVICE_GROUP|g" -e "s|REPLACE_ROOT|$ROOT|g" \
  "$ROOT/systemd/sic-pi4-attendance.service" | sudo tee "$SERVICE_PATH" >/dev/null
sudo usermod -aG video "$SERVICE_USER"
sudo systemctl daemon-reload
sudo systemctl enable sic-pi4-attendance.service
echo "Installed and enabled $SERVICE_PATH. Reboot once for video-group membership before starting it."

