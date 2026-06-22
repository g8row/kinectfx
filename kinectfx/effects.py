"""
Depth-driven visual effects for KinectFX.

Every effect here makes use of the Kinect's depth data (use Source = Depth).
Each takes a single-channel float image (`gray`, 0..1, where 1 = nearest /
0 = far/invalid) plus a dict of parameters, and returns a BGR uint8 image.

3D effects (Point Cloud, Depth Mesh) expose `yaw` / `pitch` / `distance`
parameters so the camera can be orbited from the GUI viewfinder with the mouse.

Add an effect by subclassing `Effect` and appending it to `EFFECTS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

COLORMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET,
    "Ocean": cv2.COLORMAP_OCEAN,
    "Hot": cv2.COLORMAP_HOT,
    "Bone": cv2.COLORMAP_BONE,
    "Twilight": cv2.COLORMAP_TWILIGHT_SHIFTED,
}
COLORMAP_NAMES = list(COLORMAPS.keys())


@dataclass
class Param:
    key: str
    label: str
    kind: str  # "float" | "int" | "choice" | "bool"
    default: object
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    choices: List[str] = field(default_factory=list)


def _u8(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def _apply_cmap(gray: np.ndarray, cmap_name: str) -> np.ndarray:
    return cv2.applyColorMap(_u8(gray), COLORMAPS.get(cmap_name, cv2.COLORMAP_INFERNO))


def _normals(gray: np.ndarray, strength: float) -> np.ndarray:
    """Surface normals from a depth map (smoothed), as an HxWx3 unit-vector field."""
    d = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
    nz = np.ones_like(d) / max(1e-3, strength)
    n = np.stack([-gx, -gy, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-6, None)


def _project(z0, gx, gy, W, H, depth_scale, yaw, pitch, distance):
    """Rotate a depth grid (yaw about Y, pitch about X) and perspective-project."""
    nx = gx - W / 2
    ny = gy - H / 2
    zc = (1.0 - z0) * depth_scale * 420.0  # world depth, near = small
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    xr = nx * cy_ + zc * sy_
    zt = -nx * sy_ + zc * cy_
    yr = ny * cp - zt * sp
    zr = ny * sp + zt * cp
    denom = np.clip(zr + distance, 1.0, None)
    f = 560.0
    sx = xr * f / denom + W / 2
    sy = yr * f / denom + H / 2
    return sx, sy, zr


class Effect:
    name: str = "base"
    params: List[Param] = []
    source_hint: str = "Depth"  # GUI auto-selects this source when chosen
    texture: str = None  # "rgb" or "ir": pipeline also supplies that frame to apply()

    def defaults(self) -> Dict[str, object]:
        return {p.key: p.default for p in self.params}

    def apply(self, gray: np.ndarray, p: Dict[str, object]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class PointCloud(Effect):
    """Project the depth map into a 3D point cloud you can orbit with the mouse.

    Near/far sliders clip by distance, so the background drops away
    (depth-keyed background removal for free).
    """

    name = "Point Cloud"
    params = [
        Param("cmap", "Colormap", "choice", "Turbo", choices=COLORMAP_NAMES),
        Param("near", "Near clip", "float", 0.08, 0.0, 1.0, 0.01),
        Param("far", "Far clip", "float", 1.0, 0.0, 1.0, 0.01),
        Param("depth", "Depth scale", "float", 1.0, 0.1, 3.0, 0.05),
        Param("density", "Density", "int", 3, 1, 8, 1),
        Param("point", "Point size", "int", 2, 1, 6, 1),
        Param("yaw", "Yaw °", "float", 0.0, -180.0, 180.0, 1.0),
        Param("pitch", "Pitch °", "float", 0.0, -89.0, 89.0, 1.0),
        Param("distance", "Distance", "float", 650.0, 250.0, 1500.0, 10.0),
        Param("spin", "Auto-spin", "float", 0.0, -2.0, 2.0, 0.05),
    ]

    def __init__(self):
        self._phase = 0.0

    def apply(self, gray, p):
        H, W = gray.shape
        step = int(p["density"])
        gx, gy = np.meshgrid(np.arange(0, W, step, dtype=np.float32),
                             np.arange(0, H, step, dtype=np.float32))
        z0 = gray[gy.astype(int), gx.astype(int)]
        near, far = sorted((float(p["near"]), float(p["far"])))
        keep = (z0 >= near) & (z0 <= far)

        self._phase += float(p["spin"]) * 0.05
        yaw = np.radians(float(p["yaw"])) + self._phase
        pitch = np.radians(float(p["pitch"]))
        sx, sy, zr = _project(z0, gx, gy, W, H, float(p["depth"]),
                              yaw, pitch, float(p["distance"]))
        sx = sx.astype(np.int32)
        sy = sy.astype(np.int32)
        valid = keep & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)

        colors = _apply_cmap(z0, p["cmap"])
        out = np.zeros((H, W, 3), np.uint8)
        order = np.argsort(-zr[valid])  # painter's: far first, near on top
        out[sy[valid][order], sx[valid][order]] = colors[valid][order]
        ps = int(p["point"])
        if ps > 1:
            out = cv2.dilate(out, np.ones((ps, ps), np.uint8))
        return out


class RGBCloud(Effect):
    """Point cloud textured with the real RGB camera (registered to depth).

    RGB + depth come from different sensors, so they're captured together; the
    depth is registered to the colour frame so each 3D point gets its true colour.
    Orbit with the mouse, like the other 3D effects.
    """

    name = "RGB Point Cloud"
    source_hint = "Depth"
    texture = "rgb"
    params = [
        Param("near", "Near clip", "float", 0.05, 0.0, 1.0, 0.01),
        Param("far", "Far clip", "float", 1.0, 0.0, 1.0, 0.01),
        Param("depth", "Depth scale", "float", 1.0, 0.1, 3.0, 0.05),
        Param("density", "Density", "int", 3, 1, 8, 1),
        Param("point", "Point size", "int", 2, 1, 6, 1),
        Param("gain", "Colour gain", "float", 1.0, 0.2, 3.0, 0.05),
        Param("yaw", "Yaw °", "float", 0.0, -180.0, 180.0, 1.0),
        Param("pitch", "Pitch °", "float", 0.0, -89.0, 89.0, 1.0),
        Param("distance", "Distance", "float", 650.0, 250.0, 1500.0, 10.0),
        Param("spin", "Auto-spin", "float", 0.0, -2.0, 2.0, 0.05),
    ]

    def __init__(self):
        self._phase = 0.0

    def apply(self, gray, p, rgb):
        H, W = gray.shape
        step = int(p["density"])
        gx, gy = np.meshgrid(np.arange(0, W, step, dtype=np.float32),
                             np.arange(0, H, step, dtype=np.float32))
        gxi, gyi = gx.astype(int), gy.astype(int)
        z0 = gray[gyi, gxi]
        tex = rgb[gyi, gxi].astype(np.float32) * float(p["gain"])
        tex = np.clip(tex, 0, 255).astype(np.uint8)

        near, far = sorted((float(p["near"]), float(p["far"])))
        keep = (z0 >= near) & (z0 <= far)

        self._phase += float(p["spin"]) * 0.05
        yaw = np.radians(float(p["yaw"])) + self._phase
        pitch = np.radians(float(p["pitch"]))
        sx, sy, zr = _project(z0, gx, gy, W, H, float(p["depth"]),
                              yaw, pitch, float(p["distance"]))
        sx = sx.astype(np.int32)
        sy = sy.astype(np.int32)
        valid = keep & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)

        out = np.zeros((H, W, 3), np.uint8)
        order = np.argsort(-zr[valid])
        out[sy[valid][order], sx[valid][order]] = tex[valid][order]
        ps = int(p["point"])
        if ps > 1:
            out = cv2.dilate(out, np.ones((ps, ps), np.uint8))
        return out


class RGBIRCloud(Effect):
    """Point cloud whose colour = RGB hue + IR brightness (luma swap).

    Real colours from the RGB camera, but the luminance/detail comes from the IR
    image — so the cloud stays clear in the dark (the IR projector lights the
    scene). RGB is registered to depth; IR sits in the IR-sensor frame, so use
    `ir_align` to nudge it horizontally into register. IR can't stream with depth,
    so it's refreshed every `ir_refresh` frames. Orbit with the mouse.
    """

    name = "RGB+IR Point Cloud"
    source_hint = "Depth"
    texture = "rgbir"
    params = [
        Param("near", "Near clip", "float", 0.05, 0.0, 1.0, 0.01),
        Param("far", "Far clip", "float", 1.0, 0.0, 1.0, 0.01),
        Param("depth", "Depth scale", "float", 1.0, 0.1, 3.0, 0.05),
        Param("density", "Density", "int", 3, 1, 8, 1),
        Param("point", "Point size", "int", 2, 1, 6, 1),
        Param("ir_gain", "IR brightness", "float", 1.0, 0.2, 4.0, 0.05),
        Param("saturation", "Saturation", "float", 1.4, 0.0, 3.0, 0.05),
        Param("ir_align", "IR align", "int", 0, -40, 40, 1),
        Param("ir_refresh", "IR refresh", "int", 15, 1, 60, 1),
        Param("yaw", "Yaw °", "float", 0.0, -180.0, 180.0, 1.0),
        Param("pitch", "Pitch °", "float", 0.0, -89.0, 89.0, 1.0),
        Param("distance", "Distance", "float", 650.0, 250.0, 1500.0, 10.0),
        Param("spin", "Auto-spin", "float", 0.0, -2.0, 2.0, 0.05),
    ]

    def __init__(self):
        self._phase = 0.0

    def apply(self, gray, p, tex):
        bgr, ir = tex
        shift = int(p["ir_align"])
        if shift:
            ir = np.roll(ir, shift, axis=1)
        # RGB chroma + IR luma via YCrCb luma swap
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[..., 0] = np.clip(ir.astype(np.float32) * float(p["ir_gain"]), 0, 255)
        sat = float(p["saturation"])
        if sat != 1.0:
            ycc[..., 1] = np.clip((ycc[..., 1] - 128) * sat + 128, 0, 255)
            ycc[..., 2] = np.clip((ycc[..., 2] - 128) * sat + 128, 0, 255)
        texbgr = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        H, W = gray.shape
        step = int(p["density"])
        gx, gy = np.meshgrid(np.arange(0, W, step, dtype=np.float32),
                             np.arange(0, H, step, dtype=np.float32))
        gxi, gyi = gx.astype(int), gy.astype(int)
        z0 = gray[gyi, gxi]
        col = texbgr[gyi, gxi]

        near, far = sorted((float(p["near"]), float(p["far"])))
        keep = (z0 >= near) & (z0 <= far)
        self._phase += float(p["spin"]) * 0.05
        yaw = np.radians(float(p["yaw"])) + self._phase
        pitch = np.radians(float(p["pitch"]))
        sx, sy, zr = _project(z0, gx, gy, W, H, float(p["depth"]),
                              yaw, pitch, float(p["distance"]))
        sx = sx.astype(np.int32)
        sy = sy.astype(np.int32)
        valid = keep & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)

        out = np.zeros((H, W, 3), np.uint8)
        order = np.argsort(-zr[valid])
        out[sy[valid][order], sx[valid][order]] = col[valid][order]
        ps = int(p["point"])
        if ps > 1:
            out = cv2.dilate(out, np.ones((ps, ps), np.uint8))
        return out


def _draw_runs(img, xs, ys, z0, colors, jump, thickness):
    """Draw contiguous valid runs of a projected line as anti-aliased polylines."""
    n = xs.shape[0]
    i = 0
    while i < n:
        if not np.isfinite(z0[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(z0[j + 1]) and abs(z0[j + 1] - z0[j]) < jump:
            j += 1
        if j > i:
            pts = np.stack([xs[i:j + 1], ys[i:j + 1]], axis=1).astype(np.int32)
            c = colors[(i + j) // 2]
            cv2.polylines(img, [pts], False, (int(c[0]), int(c[1]), int(c[2])),
                          thickness, cv2.LINE_AA)
        i = j + 1


class DepthMesh(Effect):
    """Depth as a rotating wireframe surface (orbit with the mouse)."""

    name = "Depth Mesh"
    params = [
        Param("cmap", "Colormap", "choice", "Viridis", choices=COLORMAP_NAMES),
        Param("near", "Near clip", "float", 0.08, 0.0, 1.0, 0.01),
        Param("far", "Far clip", "float", 1.0, 0.0, 1.0, 0.01),
        Param("depth", "Depth scale", "float", 1.0, 0.1, 3.0, 0.05),
        Param("grid", "Grid step", "int", 10, 4, 24, 1),
        Param("jump", "Edge cutoff", "float", 0.12, 0.02, 0.5, 0.01),
        Param("yaw", "Yaw °", "float", 0.0, -180.0, 180.0, 1.0),
        Param("pitch", "Pitch °", "float", 20.0, -89.0, 89.0, 1.0),
        Param("distance", "Distance", "float", 700.0, 250.0, 1500.0, 10.0),
        Param("spin", "Auto-spin", "float", 0.0, -2.0, 2.0, 0.05),
    ]

    def __init__(self):
        self._phase = 0.0

    def apply(self, gray, p):
        H, W = gray.shape
        step = int(p["grid"])
        gx, gy = np.meshgrid(np.arange(0, W, step, dtype=np.float32),
                             np.arange(0, H, step, dtype=np.float32))
        z0 = gray[gy.astype(int), gx.astype(int)]
        near, far = sorted((float(p["near"]), float(p["far"])))
        z0 = np.where((z0 >= near) & (z0 <= far), z0, np.nan)

        self._phase += float(p["spin"]) * 0.05
        yaw = np.radians(float(p["yaw"])) + self._phase
        pitch = np.radians(float(p["pitch"]))
        z_filled = np.nan_to_num(z0, nan=0.0)
        sx, sy, _ = _project(z_filled, gx, gy, W, H, float(p["depth"]),
                             yaw, pitch, float(p["distance"]))
        colors = _apply_cmap(np.nan_to_num(z0), p["cmap"])
        jump = float(p["jump"])
        out = np.zeros((H, W, 3), np.uint8)
        rows, cols = z0.shape
        for r in range(rows):
            _draw_runs(out, sx[r], sy[r], z0[r], colors[r], jump, 1)
        for c in range(cols):
            _draw_runs(out, sx[:, c], sy[:, c], z0[:, c], colors[:, c], jump, 1)
        return out


class ReliefLight(Effect):
    """Lambert-shade the depth surface with a movable light — a 3D sculpture look."""

    name = "Relief Light"
    params = [
        Param("cmap", "Tint", "choice", "Bone", choices=COLORMAP_NAMES),
        Param("strength", "Relief", "float", 0.15, 0.02, 1.0, 0.01),
        Param("azimuth", "Light °", "float", 135.0, 0.0, 360.0, 1.0),
        Param("elevation", "Elevation °", "float", 45.0, 5.0, 90.0, 1.0),
        Param("ambient", "Ambient", "float", 0.2, 0.0, 1.0, 0.02),
        Param("tintamt", "Tint amount", "float", 0.5, 0.0, 1.0, 0.05),
    ]

    def apply(self, gray, p):
        n = _normals(gray, float(p["strength"]))
        az = np.radians(float(p["azimuth"]))
        el = np.radians(float(p["elevation"]))
        light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                         np.float32)
        lamb = np.clip((n * light).sum(-1), 0, 1)
        shade = np.clip(float(p["ambient"]) + (1 - float(p["ambient"])) * lamb, 0, 1)
        tint = _apply_cmap(gray, p["cmap"]).astype(np.float32) / 255.0
        ta = float(p["tintamt"])
        base = (1 - ta) + ta * tint  # white-ish to tinted
        out = base * shade[..., None]
        return np.clip(out * 255, 0, 255).astype(np.uint8)


class NormalMap(Effect):
    """Visualise the depth surface normals as an RGB normal map."""

    name = "Normal Map"
    params = [
        Param("strength", "Relief", "float", 0.15, 0.02, 1.0, 0.01),
        Param("smooth", "Smooth", "int", 2, 0, 12, 1),
    ]

    def apply(self, gray, p):
        k = int(p["smooth"])
        g = cv2.GaussianBlur(gray, (0, 0), k) if k > 0 else gray
        n = _normals(g, float(p["strength"]))
        rgb = (n * 0.5 + 0.5)  # [-1,1] -> [0,1], (x,y,z)
        bgr = rgb[..., ::-1]   # store as BGR
        invalid = gray <= 1e-3
        out = (bgr * 255).astype(np.uint8)
        out[invalid] = (40, 30, 30)
        return out


class DepthFog(Effect):
    """Colour by distance with an atmospheric fog falloff."""

    name = "Depth Fog"
    params = [
        Param("cmap", "Colormap", "choice", "Ocean", choices=COLORMAP_NAMES),
        Param("near", "Near clip", "float", 0.05, 0.0, 1.0, 0.01),
        Param("far", "Far clip", "float", 1.0, 0.0, 1.0, 0.01),
        Param("density", "Fog density", "float", 2.0, 0.0, 6.0, 0.1),
        Param("gamma", "Gamma", "float", 1.0, 0.2, 3.0, 0.05),
    ]

    def apply(self, gray, p):
        near, far = sorted((float(p["near"]), float(p["far"])))
        g = np.power(np.clip(gray, 0, 1), p["gamma"])
        colored = _apply_cmap(g, p["cmap"]).astype(np.float32)
        dist = 1.0 - g  # 0 near .. 1 far
        fog = np.exp(-float(p["density"]) * (1.0 - dist))
        keep = ((gray >= near) & (gray <= far)).astype(np.float32)
        out = colored * fog[..., None] * keep[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class ScanPlane(Effect):
    """A bright plane sweeps through depth, lighting up a moving slice (LIDAR-style)."""

    name = "Scan Plane"
    params = [
        Param("cmap", "Colormap", "choice", "Inferno", choices=COLORMAP_NAMES),
        Param("speed", "Sweep speed", "float", 0.5, 0.0, 3.0, 0.05),
        Param("thickness", "Slice width", "float", 0.06, 0.01, 0.4, 0.01),
        Param("ambient", "Ambient", "float", 0.18, 0.0, 1.0, 0.02),
        Param("manual", "Manual pos", "float", -1.0, -1.0, 1.0, 0.01),
    ]

    def __init__(self):
        self._phase = 0.0

    def apply(self, gray, p):
        manual = float(p["manual"])
        if manual >= 0.0:
            pos = manual
        else:
            self._phase += float(p["speed"]) * 0.02
            pos = 0.5 + 0.5 * np.sin(self._phase)  # ping-pong 0..1
        w = float(p["thickness"])
        glow = np.exp(-((gray - pos) / w) ** 2)
        glow[gray <= 1e-3] = 0.0
        base = _apply_cmap(gray, p["cmap"]).astype(np.float32)
        out = base * float(p["ambient"]) + base * glow[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class Topographic(Effect):
    """Quantise depth into bands and draw the contour boundaries."""

    name = "Topographic"
    params = [
        Param("cmap", "Colormap", "choice", "Turbo", choices=COLORMAP_NAMES),
        Param("bands", "Bands", "int", 16, 2, 64, 1),
        Param("thickness", "Line width", "int", 1, 1, 6, 1),
        Param("fill", "Fill opacity", "float", 0.15, 0.0, 1.0, 0.05),
        Param("gamma", "Gamma", "float", 1.0, 0.2, 3.0, 0.05),
    ]

    def apply(self, gray, p):
        g = np.power(np.clip(gray, 0, 1), p["gamma"])
        bands = int(p["bands"])
        q = np.clip(np.floor(g * bands).astype(np.int32), 0, bands - 1)
        grad = cv2.morphologyEx(q.astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        edges = (grad > 0).astype(np.uint8)
        th = int(p["thickness"])
        if th > 1:
            edges = cv2.dilate(edges, np.ones((th, th), np.uint8))
        level = q.astype(np.float32) / max(bands - 1, 1)
        colored = _apply_cmap(level, p["cmap"]).astype(np.float32)
        out = colored * float(p["fill"])
        out[edges > 0] = colored[edges > 0]
        return np.clip(out, 0, 255).astype(np.uint8)


class DepthSlice(Effect):
    """Keep a near/far slice as a glowing silhouette."""

    name = "Depth Slice"
    params = [
        Param("cmap", "Colormap", "choice", "Magma", choices=COLORMAP_NAMES),
        Param("near", "Near", "float", 0.1, 0.0, 1.0, 0.01),
        Param("far", "Far", "float", 0.7, 0.0, 1.0, 0.01),
        Param("edge", "Outline", "float", 1.0, 0.0, 1.0, 0.05),
        Param("feather", "Feather", "int", 5, 0, 41, 2),
    ]

    def apply(self, gray, p):
        near, far = sorted((float(p["near"]), float(p["far"])))
        mask = ((gray >= near) & (gray <= far)).astype(np.float32)
        k = int(p["feather"])
        if k > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), k)
        colored = _apply_cmap(gray, p["cmap"]).astype(np.float32)
        out = colored * mask[..., None]
        if p["edge"] > 0:
            m8 = (mask > 0.5).astype(np.uint8) * 255
            outline = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            e = 255 * p["edge"]
            out[outline > 0] = (e, e, e)
        return np.clip(out, 0, 255).astype(np.uint8)


class ThermalSilhouette(Effect):
    """The Predator look: the depth-keyed body glows in false colour, dead black elsewhere.

    The IR image works in the dark and has real surface detail, so it's the
    picture; the depth map defines the near/far slice that isolates the person
    (depth-keyed background removal). IR and depth come from the same IR-sensor
    optics on Kinect v1, so they're pixel-aligned without registration.
    """

    name = "Thermal Silhouette"
    source_hint = "Depth"
    texture = "ir"
    params = [
        Param("cmap", "Palette", "choice", "Inferno", choices=COLORMAP_NAMES),
        Param("near", "Near", "float", 0.08, 0.0, 1.0, 0.01),
        Param("far", "Far", "float", 0.9, 0.0, 1.0, 0.01),
        Param("gain", "IR gain", "float", 1.3, 0.3, 4.0, 0.05),
        Param("glow", "Edge glow", "float", 0.7, 0.0, 2.0, 0.05),
        Param("feather", "Feather", "int", 4, 0, 25, 1),
    ]

    def apply(self, gray, p, ir):
        near, far = sorted((float(p["near"]), float(p["far"])))
        # depth mask isolates the person; feather softens the rim
        mask = ((gray >= near) & (gray <= far)).astype(np.float32)
        k = int(p["feather"])
        if k > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), k)
        # IR is the picture — boost its contrast, map to false colour
        irf = np.clip(ir.astype(np.float32) * float(p["gain"]), 0, 255) / 255.0
        colored = _apply_cmap(irf, p["cmap"]).astype(np.float32)
        out = colored * mask[..., None]
        # bloom the rim so the body looks like it's emitting light
        g = float(p["glow"])
        if g > 0:
            bloom = cv2.GaussianBlur(mask, (0, 0), 12)
            out += colored * (bloom * g)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class SlitScan(Effect):
    """Each pixel samples a different past frame, keyed by depth.

    Near pixels show the present; far pixels show the deeper past — so waving
    your hand drags a temporal wake through space. Needs a rolling buffer of
    recent depth frames; the GUI passes `self._frame_window` in via the param
    dict under key "frame_window" (managed by the effect, not a slider).
    """

    name = "Slit Scan"
    source_hint = "Depth"
    params = [
        Param("cmap", "Colormap", "choice", "Turbo", choices=COLORMAP_NAMES),
        Param("delay", "Time depth", "float", 0.8, 0.0, 1.5, 0.02),
        Param("speed", "Flow speed", "float", 1.0, 0.2, 4.0, 0.05),
        Param("tint", "Tint", "float", 0.4, 0.0, 1.0, 0.05),
    ]

    def __init__(self):
        self._window = None  # deque of recent gray frames, filled lazily
        self._idx = 0

    def apply(self, gray, p):
        from collections import deque
        maxlen = 48
        if self._window is None:
            self._window = deque(maxlen=maxlen)
        self._window.append(gray.copy())
        # how deep into the past to reach (0 = now, 1 = oldest)
        delay = float(p["delay"]) * maxlen
        # near (gray→1) = now, far (gray→0) = past
        n = len(self._window)
        # map each pixel's depth to a frame index into the past
        # near→0 frames back, far→`delay` frames back
        back = np.clip((1.0 - gray) * delay, 0, max(n - 1, 0)).astype(np.int32)
        # assemble the time-displaced image by indexing the buffer
        past = np.stack(list(self._window))  # (n, H, W)
        yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        # clamp index to what we have
        back = np.clip(back, 0, n - 1)
        displaced = past[back, yy, xx]
        colored = _apply_cmap(displaced, p["cmap"]).astype(np.float32)
        now_colored = _apply_cmap(gray, p["cmap"]).astype(np.float32)
        ta = float(p["tint"])
        out = now_colored * (1 - ta) + colored * ta
        return np.clip(out, 0, 255).astype(np.uint8)


class NightVision(Effect):
    """Green-phosphor night-vision goggle look over the IR image.

    IR + depth come together: IR is the picture (works in the dark), depth gives
    optional auto-gain (near things brighter) and a subtle vignette. Bloom + scan
    lines + grain sell the CRT phosphor aesthetic.
    """

    name = "Night Vision"
    source_hint = "IR"
    params = [
        Param("gain", "Gain", "float", 1.4, 0.3, 4.0, 0.05),
        Param("bloom", "Bloom", "float", 0.8, 0.0, 2.0, 0.05),
        Param("scan", "Scan lines", "float", 0.4, 0.0, 1.0, 0.05),
        Param("grain", "Grain", "float", 0.25, 0.0, 1.0, 0.05),
        Param("vignette", "Vignette", "float", 0.5, 0.0, 1.0, 0.05),
        Param("invert", "Invert", "bool", False),
    ]

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def apply(self, gray, p):
        g = np.clip(gray * float(p["gain"]), 0, 1)
        if p["invert"]:
            g = 1.0 - g
        # green phosphor: drive green hard, sprinkle red/blue for tint depth
        out = np.zeros((*g.shape, 3), np.float32)
        out[..., 1] = g
        out[..., 0] = g * 0.15
        out[..., 2] = g * 0.05
        # bloom: blur the bright parts and screen-blend back
        b = float(p["bloom"])
        if b > 0:
            bright = cv2.GaussianBlur(g, (0, 0), 8)
            out = out + (bright ** 2)[..., None] * np.array([0.1, 1.0, 0.2]) * b
        # scan lines: darken every other row
        sl = float(p["scan"])
        if sl > 0:
            lines = np.ones(g.shape[0], np.float32)
            lines[1::2] = 1.0 - sl
            out *= lines[:, None, None]
        # grain
        gr = float(p["grain"])
        if gr > 0:
            noise = self._rng.normal(0, gr * 0.25, g.shape).astype(np.float32)
            out += noise[..., None] * np.array([0.2, 1.0, 0.2])
        # vignette
        v = float(p["vignette"])
        if v > 0:
            h, w = g.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            vig = np.clip(1.0 - d * v, 0, 1)
            out *= vig[..., None]
        return np.clip(out * 255, 0, 255).astype(np.uint8)


# ASCII ramp: space (darkest) to a dense glyph (brightest). Bright IR = a body.
_ASCII_RAMP = " .:-=+*#%@"


class AsciiArt(Effect):
    """Terminal/matrix aesthetic over the IR image.

    Downsamples IR into a grid of cells and draws a glyph per cell from a
    brightness ramp. Colour tints towards green for a "hacker" feel. IR is the
    right source here: it's already monochrome and crisp in the dark, so the
    brightness→glyph mapping reads cleanly.
    """

    name = "ASCII Art"
    source_hint = "IR"
    params = [
        Param("cell", "Cell size", "int", 10, 6, 24, 1),
        Param("gain", "Gain", "float", 1.3, 0.5, 4.0, 0.05),
        Param("tint", "Green tint", "float", 0.8, 0.0, 1.0, 0.05),
        Param("bg", "Background", "float", 0.05, 0.0, 1.0, 0.05),
    ]

    def apply(self, gray, p):
        H, W = gray.shape
        cell = int(p["cell"])
        ch, cw = H // cell, W // cell
        # sample IR brightness per cell
        small = cv2.resize(gray, (cw, ch), interpolation=cv2.INTER_AREA)
        small = np.clip(small * float(p["gain"]), 0, 1)
        bg = float(p["bg"])
        tint = float(p["tint"])
        out = np.full((H, W, 3), bg * 255, np.uint8)
        ramp = _ASCII_RAMP
        # glyph colour: white → green as tint increases
        base = np.array([int(255 * (1 - tint * 0.7)),
                         255,
                         int(255 * (1 - tint))])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for y in range(ch):
            for x in range(cw):
                v = small[y, x]
                if v < 0.04:
                    continue
                ch_ = ramp[int(np.clip(v * (len(ramp) - 1), 0, len(ramp) - 1))]
                px = x * cell
                py = y * cell + cell - 1  # baseline at cell bottom
                cv2.putText(out, ch_, (px, py), font,
                            cell / 22.0, tuple(int(c) for c in base), 1, cv2.LINE_AA)
        return out


EFFECTS: List[Effect] = [
    PointCloud(),
    RGBCloud(),
    RGBIRCloud(),
    DepthMesh(),
    ReliefLight(),
    NormalMap(),
    DepthFog(),
    ScanPlane(),
    Topographic(),
    DepthSlice(),
    ThermalSilhouette(),
    SlitScan(),
    NightVision(),
    AsciiArt(),
]

EFFECTS_BY_NAME: Dict[str, Effect] = {e.name: e for e in EFFECTS}
EFFECT_NAMES = [e.name for e in EFFECTS]
