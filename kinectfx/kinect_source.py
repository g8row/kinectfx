"""
Kinect v1 (Xbox 360) capture via ctypes bindings to libfreenect's *sync* API.

Works on macOS, Windows and Linux: we locate the platform's freenect_sync
library (.dylib / .dll / .so) and bind its stable C functions via ctypes,
avoiding the Cython `freenect` module (painful to build everywhere):

    int  freenect_sync_get_video(void **video, uint32_t *ts, int idx, int fmt);
    int  freenect_sync_get_depth(void **depth, uint32_t *ts, int idx, int fmt);
    void freenect_sync_stop(void);

These spin up libfreenect's USB event loop on first call and then hand back the
latest frame on each subsequent call — perfect for a polling capture loop.

If no Kinect / library is available the module falls back to a synthetic
animated source so the GUI is still usable for development.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# libfreenect enum values (from libfreenect.h)
# ---------------------------------------------------------------------------
FREENECT_VIDEO_RGB = 0
FREENECT_VIDEO_IR_8BIT = 2

FREENECT_DEPTH_11BIT = 0       # raw disparity, 0..2047
FREENECT_DEPTH_REGISTERED = 4  # depth in mm, aligned to the RGB camera frame

# Medium resolution is what the sync API uses by default.
WIDTH, HEIGHT = 640, 480

# Retries to ride out the brief error window when freenect_sync restarts a
# stream (happens when switching source / video format).
RETRIES = 8
RETRY_DELAY = 0.05

# Override hooks (any platform):
#   KINECTFX_FREENECT  full path to the freenect_sync library
#   KINECTFX_LIB_DIR   directory holding the library + its dependent DLLs/.so
_ENV_LIB = "KINECTFX_FREENECT"
_ENV_LIBDIR = "KINECTFX_LIB_DIR"


def _candidate_names():
    """Platform-specific library file names / well-known locations."""
    if sys.platform == "win32":
        return ["freenect_sync.dll", "libfreenect_sync.dll"]
    if sys.platform == "darwin":
        return [
            "/opt/homebrew/lib/libfreenect_sync.dylib",  # Homebrew (Apple Silicon)
            "/opt/homebrew/lib/libfreenect_sync.0.dylib",
            "/usr/local/lib/libfreenect_sync.dylib",      # Homebrew (Intel)
            "/usr/local/lib/libfreenect_sync.0.dylib",
            "libfreenect_sync.dylib",
        ]
    return [  # Linux / other
        "libfreenect_sync.so",
        "libfreenect_sync.so.0",
        "libfreenect_sync.so.0.7",
        "/usr/local/lib/libfreenect_sync.so",
        "/usr/lib/libfreenect_sync.so",
    ]


def _extra_dirs():
    """Directories to search for the library and (on Windows) its deps."""
    dirs = []
    if os.environ.get(_ENV_LIBDIR):
        dirs.append(os.environ[_ENV_LIBDIR])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs.append(os.path.join(repo, "libs"))  # bundle DLLs/.so here if you like
    return [d for d in dirs if os.path.isdir(d)]


def _load_library():
    # On Windows, dependent DLLs (libusb-1.0.dll, freenect.dll, pthread*.dll)
    # must be discoverable; register our extra dirs with the DLL loader.
    if sys.platform == "win32":
        for d in _extra_dirs():
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass

    candidates = []
    if os.environ.get(_ENV_LIB):
        candidates.append(os.environ[_ENV_LIB])
    found = ctypes.util.find_library("freenect_sync")
    if found:
        candidates.append(found)
    candidates += _candidate_names()
    for d in _extra_dirs():  # explicit name-in-dir combos
        for name in _candidate_names():
            if not os.path.isabs(name):
                candidates.append(os.path.join(d, name))

    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


class KinectError(RuntimeError):
    pass


@dataclass
class FrameInfo:
    width: int
    height: int


class KinectSource:
    """Thin wrapper over freenect_sync. Thread-affine: drive from one thread."""

    def __init__(self):
        self._lib = _load_library()
        self.available = self._lib is not None
        if self.available:
            self._configure_signatures()

    # -- library plumbing ---------------------------------------------------
    def _configure_signatures(self):
        lib = self._lib
        lib.freenect_sync_get_video.restype = ctypes.c_int
        lib.freenect_sync_get_video.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.freenect_sync_get_depth.restype = ctypes.c_int
        lib.freenect_sync_get_depth.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.freenect_sync_stop.restype = None
        lib.freenect_sync_stop.argtypes = []

    def _get_video(self, fmt, channels, dtype):
        ptr = ctypes.c_void_p()
        ts = ctypes.c_uint32()
        # freenect_sync restarts the stream when the format changes (e.g. when
        # switching source), and can briefly return an error during that
        # restart. Retry a few times before giving up.
        rc = -1
        for attempt in range(RETRIES):
            rc = self._lib.freenect_sync_get_video(
                ctypes.byref(ptr), ctypes.byref(ts), 0, fmt
            )
            if rc == 0 and ptr.value:
                break
            time.sleep(RETRY_DELAY)
        if rc != 0 or not ptr.value:
            raise KinectError(f"freenect_sync_get_video failed (rc={rc})")
        count = WIDTH * HEIGHT * channels
        itemsize = np.dtype(dtype).itemsize
        buf = (ctypes.c_char * (count * itemsize)).from_address(ptr.value)
        arr = np.frombuffer(buf, dtype=dtype).copy()
        if channels == 1:
            return arr.reshape(HEIGHT, WIDTH)
        return arr.reshape(HEIGHT, WIDTH, channels)

    def _get_depth_raw(self, fmt=FREENECT_DEPTH_11BIT):
        ptr = ctypes.c_void_p()
        ts = ctypes.c_uint32()
        rc = -1
        for attempt in range(RETRIES):
            rc = self._lib.freenect_sync_get_depth(
                ctypes.byref(ptr), ctypes.byref(ts), 0, fmt
            )
            if rc == 0 and ptr.value:
                break
            time.sleep(RETRY_DELAY)
        if rc != 0 or not ptr.value:
            raise KinectError(f"freenect_sync_get_depth failed (rc={rc})")
        count = WIDTH * HEIGHT
        buf = (ctypes.c_char * (count * 2)).from_address(ptr.value)
        return np.frombuffer(buf, dtype=np.uint16).copy().reshape(HEIGHT, WIDTH)

    # -- public API ---------------------------------------------------------
    def get_ir(self) -> np.ndarray:
        """640x480 uint8 IR image."""
        if not self.available:
            return _synthetic("ir")
        return self._get_video(FREENECT_VIDEO_IR_8BIT, 1, np.uint8)

    def get_rgb(self) -> np.ndarray:
        """640x480x3 uint8 RGB image."""
        if not self.available:
            return _synthetic("rgb")
        return self._get_video(FREENECT_VIDEO_RGB, 3, np.uint8)

    def get_depth(self) -> np.ndarray:
        """640x480 uint16 raw disparity (0..2047, 2047 = no reading)."""
        if not self.available:
            return _synthetic("depth")
        return self._get_depth_raw()

    def get_depth_registered(self) -> np.ndarray:
        """640x480 uint16 depth in mm, aligned to the RGB frame (0 = no reading).

        Pairs with get_rgb(): depth[y, x] corresponds to rgb[y, x], so the colour
        camera can texture the depth geometry. RGB + depth use different sensors,
        so this works simultaneously (unlike IR + depth)."""
        if not self.available:
            d = _synthetic("depth").astype(np.float32) / 2047.0
            return (d * 4000.0).astype(np.uint16)
        return self._get_depth_raw(FREENECT_DEPTH_REGISTERED)

    def stop(self):
        if self.available:
            try:
                self._lib.freenect_sync_stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Synthetic fallback so the app runs without hardware.
# ---------------------------------------------------------------------------
def _synthetic(kind: str) -> np.ndarray:
    t = time.time()
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    cx = WIDTH / 2 + 160 * np.cos(t * 0.7)
    cy = HEIGHT / 2 + 110 * np.sin(t * 0.9)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    wave = 0.5 + 0.5 * np.sin(r * 0.05 - t * 4.0)
    if kind == "depth":
        d = (2047 * (1.0 - wave)).astype(np.uint16)
        return d
    g = (wave * 255).astype(np.uint8)
    if kind == "rgb":
        return np.stack([g, np.roll(g, 30, 1), np.roll(g, 60, 1)], axis=-1)
    return g


if __name__ == "__main__":  # smoke test
    src = KinectSource()
    print("library available:", src.available)
    try:
        ir = src.get_ir()
        print("IR frame:", ir.shape, ir.dtype, "min/max", ir.min(), ir.max())
    except Exception as e:
        print("IR error:", e)
    src.stop()
