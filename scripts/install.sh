#!/usr/bin/env bash
set -euo pipefail

source /etc/os-release
[[ "$(uname -m)" == "aarch64" ]] || { echo "Raspberry Pi OS 64-bit (aarch64) is required." >&2; exit 1; }
[[ "${ID:-}" == "debian" && "${VERSION_CODENAME:-}" == "trixie" ]] || {
  echo "Raspberry Pi OS based on Debian 13 (Trixie) is required." >&2
  exit 1
}
grep -aq "Raspberry Pi 4" /proc/device-tree/model || {
  echo "Raspberry Pi 4 hardware is required." >&2
  exit 1
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo apt update
sudo apt install -y build-essential cmake git libgl1 libglib2.0-0t64 libopenblas-dev \
  python3-dev python3-pip python3-tk python3-venv v4l-utils

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT[test]"
"$ROOT/.venv/bin/python" -c 'import cv2, tkinter; from PIL import Image, ImageTk'

SERVICE_USER="${SUDO_USER:-$USER}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
sudo install -d -m 750 -o "$SERVICE_USER" -g "$SERVICE_GROUP" /var/lib/sic-attendance
sudo install -d -m 750 -o "$SERVICE_USER" -g "$SERVICE_GROUP" /opt/sic-attendance/models
sudo install -d -m 700 -o "$SERVICE_USER" -g "$SERVICE_GROUP" /opt/sic-attendance/secrets

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  chmod 600 "$ROOT/.env"
fi

echo "Dependencies and runtime directories installed."
