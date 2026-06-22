"""
Virtual camera output via pyvirtualcam.

pyvirtualcam pushes BGR/RGB frames to a virtual camera that apps like Discord,
Zoom and Meet can select as a webcam:
  * macOS   -> OBS Virtual Camera (CoreMediaIO); OBS 30+ installed and its
               virtual camera started once to register the system extension.
  * Windows -> OBS Virtual Camera (DirectShow) or Unity Capture.
  * Linux   -> v4l2loopback.

This wrapper never raises into the capture loop: if the backend is missing it
reports an error string and simply does nothing, so the GUI preview still runs.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    _HAVE = True
except Exception as e:  # pragma: no cover
    pyvirtualcam = None
    PixelFormat = None
    _HAVE = False
    _IMPORT_ERR = str(e)


def _backend_help() -> str:
    if sys.platform == "win32":
        return ("On Windows: install OBS Studio, run it once and click "
                "'Start Virtual Camera' (registers the camera), then quit OBS and "
                "retry. OBS must NOT be running its virtual camera at the same time.")
    if sys.platform == "darwin":
        return ("On macOS: install OBS 30+ (brew install --cask obs), open it once, "
                "click 'Start Virtual Camera' and approve the system extension in "
                "System Settings, then quit OBS and retry.")
    return ("On Linux: load v4l2loopback "
            "(sudo modprobe v4l2loopback exclusive_caps=1), then retry.")


class VirtualCam:
    def __init__(self):
        self._cam = None
        self.width = 0
        self.height = 0
        self.error: Optional[str] = None
        self.device: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._cam is not None

    def start(self, width: int, height: int, fps: int = 30) -> bool:
        if not _HAVE:
            self.error = f"pyvirtualcam not installed: {_IMPORT_ERR}"
            return False
        self.stop()
        try:
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps, fmt=PixelFormat.BGR
            )
            self.width, self.height = width, height
            self.device = self._cam.device
            self.error = None
            return True
        except Exception as e:
            self._cam = None
            self.error = f"{e}\n\n{_backend_help()}"
            return False

    def send(self, bgr: np.ndarray):
        if self._cam is None:
            return
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            return  # size changed; caller should restart
        try:
            self._cam.send(bgr)
            self._cam.sleep_until_next_frame()
        except Exception as e:
            self.error = str(e)
            self.stop()

    def stop(self):
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
