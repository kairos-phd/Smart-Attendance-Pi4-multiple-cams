#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Warning: this package is intended for 64-bit Raspberry Pi OS (aarch64)." >&2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo apt update
sudo apt install -y build-essential cmake git libatlas-base-dev libgl1 libglib2.0-0 libopenblas-dev \
  libopencv-dev python3-dev python3-opencv python3-pip python3-venv v4l-utils

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT[test]"

sudo install -d -m 750 -o "${SUDO_USER:-$USER}" -g "${SUDO_USER:-$USER}" /var/lib/sic-attendance
sudo install -d -m 755 /opt/sic-attendance/models
sudo install -d -m 700 /opt/sic-attendance/secrets

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  chmod 600 "$ROOT/.env"
fi

echo "Installed. Edit $ROOT/.env, prepare models, then run scripts/install_systemd.sh."

