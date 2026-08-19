#!/usr/bin/env bash
set -euo pipefail

echo "== Raspberry Pi =="
uname -a
grep -E 'Model|Hardware' /proc/cpuinfo | tail -n 2 || true
echo "== Temperature and throttling =="
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
echo "== Video devices =="
v4l2-ctl --list-devices
echo "== Stable camera paths =="
find /dev/v4l/by-id -maxdepth 1 -type l -print -exec readlink -f {} \; 2>/dev/null || true
echo "== Storage =="
findmnt -T /var/lib/sic-attendance || true
df -h /var/lib/sic-attendance || true
echo "== Memory =="
free -h

