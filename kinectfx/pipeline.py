"""
Capture -> normalise -> effect -> feedback trails -> output.

The `Pipeline` owns the Kinect source and the per-frame state. The GUI writes
parameters into `Pipeline` (guarded by a lock) and the capture thread reads
them, so the UI stays responsive while frames are produced.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict

import cv2
import numpy as np

from . import effects
from .kinect_source import KinectSource

SOURCES = ["IR", "Depth", "RGB"]


@dataclass
class State:
    source: str = "Depth"
    effect: str = effects.EFFECT_NAMES[0]
    mirror: bool = True
    brightness: float = 0.0  # -1..1 added
    contrast: float = 1.0    # 0..3 multiply
    feedback: float = 0.0    # 0..0.97 trail decay
    out_w: int = 960
    out_h: int = 720
    # per-effect parameter values: {effect_name: {key: value}}
    effect_params: Dict[str, Dict[str, object]] = field(default_factory=dict)


class Pipeline:
    def __init__(self):
        self.kinect = KinectSource()
        self.lock = threading.Lock()
        self.state = State()
        for e in effects.EFFECTS:
            self.state.effect_params[e.name] = e.defaults()
        self._prev = None  # previous output frame for feedback
        self._ir_tex = None  # last IR frame (for RGB+IR texture)
        self._ir_count = 0

    # -- thread-safe state access ------------------------------------------
    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self.state, k, v)

    def set_effect_param(self, effect_name: str, key: str, value):
        with self.lock:
            self.state.effect_params.setdefault(effect_name, {})[key] = value

    def snapshot(self) -> State:
        with self.lock:
            s = self.state
            return State(
                source=s.source,
                effect=s.effect,
                mirror=s.mirror,
                brightness=s.brightness,
                contrast=s.contrast,
                feedback=s.feedback,
                out_w=s.out_w,
                out_h=s.out_h,
                effect_params={k: dict(v) for k, v in s.effect_params.items()},
            )

    # -- capture -----------------------------------------------------------
    def _grab_gray(self, source: str) -> np.ndarray:
        if source == "RGB":
            rgb = self.kinect.get_rgb()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        elif source == "Depth":
            d = self.kinect.get_depth().astype(np.float32)
            # 2047 = no/invalid reading -> push to far (0). Closer = brighter.
            d[d >= 2047] = 2047
            gray = 1.0 - (d / 2047.0)
        else:  # IR
            gray = self.kinect.get_ir().astype(np.float32) / 255.0
        return gray

    def _grab_rgb_textured(self):
        """RGB (as BGR) + registered depth normalised to 0..1 (near = 1)."""
        rgb = self.kinect.get_rgb()
        dmm = self.kinect.get_depth_registered().astype(np.float32)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR for cv2
        dmax = 4000.0  # ~4 m working range
        gray = np.where(dmm > 0, 1.0 - np.clip(dmm, 0, dmax) / dmax, 0.0).astype(np.float32)
        return gray, bgr

    def _grab_rgbir(self, refresh):
        """RGB (BGR) + registered depth + IR refreshed every `refresh` frames."""
        if self._ir_tex is None or self._ir_count % max(1, int(refresh)) == 0:
            try:
                self._ir_tex = self.kinect.get_ir()
            except Exception:
                pass
        self._ir_count += 1
        gray, bgr = self._grab_rgb_textured()
        ir = self._ir_tex if self._ir_tex is not None else (gray * 255).astype(np.uint8)
        return gray, bgr, ir

    def _grab_ir(self):
        """IR frame as uint8 (refreshed in lockstep with depth via the depth map).

        IR and depth share the IR-sensor optics on Kinect v1, so the IR image is
        already pixel-aligned to the depth map (no registration needed, unlike RGB).
        Because the Kinect can't stream IR and depth simultaneously, the IR frame
        is refreshed every `ir_refresh` frames to keep depth live.
        """
        if self._ir_tex is None or self._ir_count % 6 == 0:
            try:
                self._ir_tex = self.kinect.get_ir()
            except Exception:
                pass
        self._ir_count += 1
        d = self.kinect.get_depth().astype(np.float32)
        d[d >= 2047] = 2047
        gray = 1.0 - (d / 2047.0)
        ir = self._ir_tex if self._ir_tex is not None else (gray * 255).astype(np.uint8)
        return gray, ir

    def process(self) -> np.ndarray:
        """Produce one BGR uint8 output frame at (out_h, out_w)."""
        s = self.snapshot()
        effect = effects.EFFECTS_BY_NAME[s.effect]
        params = {**effect.defaults(), **s.effect_params.get(s.effect, {})}

        if effect.texture == "rgbir":
            gray, bgr_t, ir_t = self._grab_rgbir(params.get("ir_refresh", 15))
            if s.mirror:
                gray = np.ascontiguousarray(gray[:, ::-1])
                bgr_t = np.ascontiguousarray(bgr_t[:, ::-1])
                ir_t = np.ascontiguousarray(ir_t[:, ::-1])
            gray = np.clip((gray - 0.5) * s.contrast + 0.5 + s.brightness, 0, 1)
            bgr = effect.apply(gray, params, (bgr_t, ir_t))
        elif effect.texture == "ir":
            gray, ir_t = self._grab_ir()
            if s.mirror:
                gray = np.ascontiguousarray(gray[:, ::-1])
                ir_t = np.ascontiguousarray(ir_t[:, ::-1])
            gray = np.clip((gray - 0.5) * s.contrast + 0.5 + s.brightness, 0, 1)
            bgr = effect.apply(gray, params, ir_t)
        elif effect.texture:
            gray, tex = self._grab_rgb_textured()
            if s.mirror:
                gray = np.ascontiguousarray(gray[:, ::-1])
                tex = np.ascontiguousarray(tex[:, ::-1])
            gray = np.clip((gray - 0.5) * s.contrast + 0.5 + s.brightness, 0, 1)
            bgr = effect.apply(gray, params, tex)
        else:
            gray = self._grab_gray(s.source)
            if s.mirror:
                gray = np.ascontiguousarray(gray[:, ::-1])
            gray = np.clip((gray - 0.5) * s.contrast + 0.5 + s.brightness, 0, 1)
            bgr = effect.apply(gray, params)

        # feedback / trails (screen-blend the decayed previous frame)
        if s.feedback > 0 and self._prev is not None and self._prev.shape == bgr.shape:
            decayed = (self._prev.astype(np.float32) * s.feedback)
            bgr = np.maximum(bgr.astype(np.float32), decayed).astype(np.uint8)
        self._prev = bgr

        if (bgr.shape[1], bgr.shape[0]) != (s.out_w, s.out_h):
            bgr = cv2.resize(bgr, (s.out_w, s.out_h), interpolation=cv2.INTER_LINEAR)
        return bgr

    def close(self):
        self.kinect.stop()
