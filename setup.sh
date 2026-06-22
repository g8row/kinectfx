#!/usr/bin/env bash
# One-time setup for KinectFX on macOS (Apple Silicon).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing libfreenect (Kinect v1 driver) + libusb via Homebrew"
brew install libfreenect

if ! ls /Applications/OBS.app >/dev/null 2>&1; then
  echo "==> Installing OBS (provides the macOS virtual camera backend)"
  brew install --cask obs
else
  echo "==> OBS already installed"
fi

PY=/opt/homebrew/bin/python3.11
if [ ! -x "$PY" ]; then PY=python3; fi
echo "==> Creating virtualenv (.venv) with $PY"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cat <<'EOF'

==> Done.

Next steps:
  1. Open OBS once and click "Start Virtual Camera" (registers the macOS
     virtual-camera plugin), then quit OBS. You only need to do this once.
  2. Run the app:   ./run.sh        (or: source .venv/bin/activate && python main.py)
  3. In the app, click "Start Virtual Camera".
  4. In Discord:  Settings → Voice & Video → Camera → "OBS Virtual Camera".

EOF
