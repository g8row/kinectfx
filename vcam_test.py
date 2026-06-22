#!/usr/bin/env python3
"""
Standalone virtual-camera test — no Kinect, no GUI.

Sends an obvious animated pattern (moving bar + frame counter) to the virtual
camera so you can verify the OBS Virtual Camera plumbing works on its own.

Usage:
    source .venv/bin/activate
    python vcam_test.py [seconds]

Then open Photo Booth (or Discord → Voice & Video) and pick "OBS Virtual Camera".
You should see a green bar sweeping across with a frame counter.

IMPORTANT: quit OBS first — OBS and this script can't both drive the camera.
"""
import sys
import time

import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

W, H, FPS = 960, 720, 30
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

try:
    cam = pyvirtualcam.Camera(width=W, height=H, fps=FPS, fmt=PixelFormat.BGR)
except Exception as e:
    print("Could not open virtual camera:", e)
    print("\nFix: quit OBS, make sure you started OBS Virtual Camera once to "
          "install the system extension, then retry.")
    sys.exit(1)

print(f"Sending to: {cam.device}  ({W}x{H}@{FPS}) for {seconds:.0f}s")
print("Open Photo Booth / Discord and select 'OBS Virtual Camera'.")
n = 0
end = time.time() + seconds
while time.time() < end:
    frame = np.full((H, W, 3), 30, np.uint8)
    x = int((n * 12) % W)
    cv2.rectangle(frame, (x, 0), (x + 80, H), (0, 220, 0), -1)  # sweeping green bar
    cv2.putText(frame, f"VCAM TEST  frame {n}", (40, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cam.send(frame)
    cam.sleep_until_next_frame()
    n += 1

cam.close()
print(f"Done — sent {n} frames with no errors.")
