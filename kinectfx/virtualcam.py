"""
Virtual camera output via pyvirtualcam.

On macOS pyvirtualcam pushes BGR/RGB frames to the **OBS Virtual Camera**
(CoreMediaIO) backend, which apps like Discord, Zoom and Meet can select as a
webcam. OBS 30+ must be installed and its virtual camera must have been started
at least once so the system plugin is registered.

This wrapper never raises into the capture loop: if the backend is missing it
reports an error string and simply does nothing, so the GUI preview still runs.
"""
from __future__ import annotations

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
            self.error = (
                f"{e}\n\nOn macOS: install OBS 30+ (brew install --cask obs), open it "
                "once and click 'Start Virtual Camera', then retry."
            )
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
