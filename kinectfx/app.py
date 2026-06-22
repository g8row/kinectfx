"""
KinectFX GUI (PySide6) with a modern dark theme.

Layout: live preview on the left, a scrollable control panel on the right with
the source selector, effect selector, auto-generated sliders for the current
effect, global tone / mirror / trail controls, and a pinned Start/Stop button
for the virtual camera.

A background QThread runs the capture+effect pipeline and feeds both the preview
and the virtual camera, keeping the UI responsive. The look is pure Qt/QSS — no
Swift or native code required.
"""
from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import effects
from .pipeline import SOURCES, Pipeline
from .virtualcam import VirtualCam

ACCENT = "#00d6a4"
STYLESHEET = f"""
QWidget {{
    background: #16161c;
    color: #e6e6ea;
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
#Panel {{ background: #1c1d24; }}
QScrollArea {{ border: none; background: #1c1d24; }}
#Preview {{
    background: #0c0c10;
    border: 1px solid #2a2b35;
    border-radius: 12px;
    color: #6a6b76;
}}
#Section {{
    color: #7d7e8a;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 0 2px 0;
}}
#Title {{ font-size: 15px; font-weight: 700; }}
QLabel {{ background: transparent; }}
QComboBox {{
    background: #2a2b35;
    border: 1px solid #353641;
    border-radius: 8px;
    padding: 6px 10px;
}}
QComboBox:hover {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: #23242c;
    border: 1px solid #353641;
    border-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: #0c0c10;
    outline: none;
    padding: 4px;
}}
QSlider::groove:horizontal {{
    height: 4px; background: #34353f; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 15px; height: 15px; margin: -6px 0;
    border-radius: 8px; background: {ACCENT};
}}
QSlider::handle:horizontal:hover {{ background: #2ff0c4; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid #3a3b46; background: #2a2b35;
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QPushButton {{
    background: #2a2b35; border: 1px solid #353641;
    border-radius: 9px; padding: 9px 14px; font-weight: 600;
}}
QPushButton:hover {{ border: 1px solid {ACCENT}; }}
#Primary {{ background: {ACCENT}; color: #07120f; border: none; }}
#Primary:checked {{ background: #ff5d73; color: #1a0608; }}
QFrame#HLine {{ background: #2a2b35; max-height: 1px; border: none; }}
"""


class PreviewLabel(QtWidgets.QLabel):
    """Preview that turns mouse drag into camera orbit and the wheel into zoom."""

    orbit = QtCore.Signal(float, float)   # dyaw, dpitch (pixels dragged)
    zoomed = QtCore.Signal(float)         # wheel steps

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._last = None
        self.setMouseTracking(False)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._last = e.position()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._last is not None and (e.buttons() & QtCore.Qt.LeftButton):
            p = e.position()
            self.orbit.emit(p.x() - self._last.x(), p.y() - self._last.y())
            self._last = p

    def mouseReleaseEvent(self, e):
        self._last = None
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def wheelEvent(self, e):
        self.zoomed.emit(e.angleDelta().y() / 120.0)


class CaptureThread(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray)
    status = QtCore.Signal(str)

    def __init__(self, pipeline: Pipeline, vcam: VirtualCam):
        super().__init__()
        self.pipeline = pipeline
        self.vcam = vcam
        self._running = True

    def run(self):
        while self._running:
            try:
                bgr = self.pipeline.process()
            except Exception as e:
                self.status.emit(f"capture error: {e}")
                self.msleep(200)
                continue
            self.frame_ready.emit(bgr)
            if self.vcam.running:
                self.vcam.send(bgr)
                if self.vcam.error:
                    self.status.emit(self.vcam.error.splitlines()[0])

    def stop(self):
        self._running = False
        self.wait(1500)


class SliderRow(QtWidgets.QWidget):
    """A labelled control bound to a Param; calls `on_change(key, value)`."""

    def __init__(self, param: effects.Param, value, on_change):
        super().__init__()
        self.param = param
        self.on_change = on_change
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        label = QtWidgets.QLabel(param.label)
        label.setMinimumWidth(86)
        label.setStyleSheet("color:#b9bac4;")
        lay.addWidget(label)

        if param.kind == "choice":
            self.widget = QtWidgets.QComboBox()
            self.widget.addItems(param.choices)
            self.widget.setCurrentText(str(value))
            self.widget.currentTextChanged.connect(lambda v: on_change(param.key, v))
            lay.addWidget(self.widget, 1)
        elif param.kind == "bool":
            self.widget = QtWidgets.QCheckBox()
            self.widget.setChecked(bool(value))
            self.widget.toggled.connect(lambda v: on_change(param.key, bool(v)))
            lay.addWidget(self.widget, 1)
        else:
            self.is_float = param.kind == "float"
            self.scale = 100.0 if self.is_float else 1.0
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setMinimum(int(param.minimum * self.scale))
            self.slider.setMaximum(int(param.maximum * self.scale))
            self.slider.setValue(int(float(value) * self.scale))
            self.val_label = QtWidgets.QLabel()
            self.val_label.setMinimumWidth(42)
            self.val_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.val_label.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
            self._update_label(self.slider.value())
            self.slider.valueChanged.connect(self._changed)
            lay.addWidget(self.slider, 1)
            lay.addWidget(self.val_label)

    def _changed(self, raw):
        v = raw / self.scale if self.is_float else int(raw)
        self._update_label(raw)
        self.on_change(self.param.key, v)

    def _update_label(self, raw):
        v = raw / self.scale if self.is_float else int(raw)
        self.val_label.setText(f"{v:.2f}" if self.is_float else str(v))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KinectFX")
        self.pipeline = Pipeline()
        self.vcam = VirtualCam()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # -- preview --------------------------------------------------------
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        self.preview = PreviewLabel("starting…")
        self.preview.setObjectName("Preview")
        self.preview.setMinimumSize(640, 480)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.orbit.connect(self._orbit)
        self.preview.zoomed.connect(self._zoom)
        left.addWidget(self.preview, 1)
        self.hint = QtWidgets.QLabel("drag to orbit · scroll to zoom (3D effects)")
        self.hint.setAlignment(QtCore.Qt.AlignCenter)
        self.hint.setStyleSheet("color:#5e5f6a;")
        left.addWidget(self.hint)
        root.addLayout(left, 1)

        # -- right column: scrollable controls + pinned footer --------------
        right = QtWidgets.QWidget()
        right.setObjectName("Panel")
        right.setFixedWidth(348)
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(14, 14, 14, 14)
        rlay.setSpacing(8)
        root.addWidget(right)

        title = QtWidgets.QLabel("KinectFX")
        title.setObjectName("Title")
        rlay.addWidget(title)
        sub = QtWidgets.QLabel("IR effects → virtual camera")
        sub.setStyleSheet("color:#7d7e8a;")
        rlay.addWidget(sub)

        if not self.pipeline.kinect.available:
            warn = QtWidgets.QLabel("⚠ Kinect/libfreenect not found — synthetic source")
            warn.setStyleSheet("color:#ff5d73;")
            warn.setWordWrap(True)
            rlay.addWidget(warn)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        self.controls = QtWidgets.QVBoxLayout(content)
        self.controls.setContentsMargins(0, 0, 6, 0)
        self.controls.setSpacing(6)
        scroll.setWidget(content)
        rlay.addWidget(scroll, 1)

        self.controls.addWidget(self._section("SOURCE & EFFECT"))
        self.source_combo = self._combo("Source", SOURCES,
                                        self.pipeline.state.source, self._on_source)
        self.effect_combo = self._combo("Effect", effects.EFFECT_NAMES,
                                        self.pipeline.state.effect, self._on_effect)

        self.controls.addWidget(self._section("EFFECT CONTROLS"))
        self.effect_box = QtWidgets.QVBoxLayout()
        self.effect_box.setSpacing(4)
        self.controls.addLayout(self.effect_box)

        self.controls.addWidget(self._section("GLOBAL"))
        self._add_global_slider("Brightness", -1.0, 1.0, 0.0,
                                lambda v: self.pipeline.update(brightness=v))
        self._add_global_slider("Contrast", 0.0, 3.0, 1.0,
                                lambda v: self.pipeline.update(contrast=v))
        self._add_global_slider("Trails", 0.0, 0.97, 0.0,
                                lambda v: self.pipeline.update(feedback=v))
        self.mirror_cb = QtWidgets.QCheckBox("Mirror")
        self.mirror_cb.setChecked(self.pipeline.state.mirror)
        self.mirror_cb.toggled.connect(lambda v: self.pipeline.update(mirror=bool(v)))
        self.controls.addWidget(self.mirror_cb)
        self.controls.addStretch(1)

        # pinned footer
        line = QtWidgets.QFrame()
        line.setObjectName("HLine")
        line.setFrameShape(QtWidgets.QFrame.HLine)
        rlay.addWidget(line)
        self.vcam_btn = QtWidgets.QPushButton("Start Virtual Camera")
        self.vcam_btn.setObjectName("Primary")
        self.vcam_btn.setCheckable(True)
        self.vcam_btn.toggled.connect(self._toggle_vcam)
        rlay.addWidget(self.vcam_btn)
        self.status = QtWidgets.QLabel("ready")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#7d7e8a;")
        rlay.addWidget(self.status)

        self._build_effect_sliders(self.pipeline.state.effect)

        # -- capture thread -------------------------------------------------
        self.thread = CaptureThread(self.pipeline, self.vcam)
        self.thread.frame_ready.connect(self._show_frame)
        self.thread.status.connect(self.status.setText)
        self.thread.start()

    # -- helpers -----------------------------------------------------------
    def _section(self, title):
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("Section")
        return lbl

    def _combo(self, label, items, current, cb):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(86)
        lbl.setStyleSheet("color:#b9bac4;")
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(cb)
        row.addWidget(lbl)
        row.addWidget(combo, 1)
        self.controls.addLayout(row)
        return combo

    def _add_global_slider(self, label, lo, hi, val, cb):
        param = effects.Param(label.lower(), label, "float", val, lo, hi, 0.01)
        self.controls.addWidget(SliderRow(param, val, lambda k, v: cb(v)))

    def _build_effect_sliders(self, effect_name):
        while self.effect_box.count():
            item = self.effect_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.effect_rows = {}
        effect = effects.EFFECTS_BY_NAME[effect_name]
        values = self.pipeline.state.effect_params.get(effect_name, effect.defaults())
        for p in effect.params:
            row = SliderRow(
                p, values.get(p.key, p.default),
                lambda k, v, e=effect_name: self.pipeline.set_effect_param(e, k, v),
            )
            self.effect_rows[p.key] = row
            self.effect_box.addWidget(row)
        has_cam = "yaw" in self.effect_rows
        self.hint.setVisible(has_cam)

    # -- slots -------------------------------------------------------------
    def _on_source(self, src):
        self.pipeline.update(source=src)

    def _on_effect(self, name):
        self.pipeline.update(effect=name)
        hint = effects.EFFECTS_BY_NAME[name].source_hint
        if hint and self.source_combo.currentText() != hint:
            self.source_combo.setCurrentText(hint)  # triggers _on_source
        self._build_effect_sliders(name)

    def _nudge(self, key, delta):
        """Move a slider of the current effect by `delta` param-units (mouse camera)."""
        row = self.effect_rows.get(key)
        if row is None or not hasattr(row, "slider"):
            return
        s = row.slider
        s.setValue(int(round(s.value() + delta * row.scale)))  # clamps + updates pipeline

    def _orbit(self, dx, dy):
        self._nudge("yaw", dx * 0.4)
        self._nudge("pitch", -dy * 0.4)

    def _zoom(self, steps):
        self._nudge("distance", -steps * 40.0)  # wheel up = move closer

    def _toggle_vcam(self, on):
        if on:
            s = self.pipeline.snapshot()
            ok = self.vcam.start(s.out_w, s.out_h, 30)
            if ok:
                self.vcam_btn.setText("Stop Virtual Camera")
                self.status.setText(f"● live → {self.vcam.device}")
            else:
                self.vcam_btn.setChecked(False)
                self.status.setText(self.vcam.error or "failed to start virtual cam")
        else:
            self.vcam.stop()
            self.vcam_btn.setText("Start Virtual Camera")
            self.status.setText("virtual cam stopped")

    @QtCore.Slot(np.ndarray)
    def _show_frame(self, bgr):
        rgb = bgr[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.preview.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def closeEvent(self, event):
        self.thread.stop()
        self.vcam.stop()
        self.pipeline.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.resize(1320, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
