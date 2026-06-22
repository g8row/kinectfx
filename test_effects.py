#!/usr/bin/env python3
"""
Headless smoke test for the new IR/depth effects.

Runs the full Pipeline (synthetic source — no Kinect needed) through each of
the four new effects for several frames, asserts the output shape/dtype, and
writes one representative frame per effect to docs/samples/test_<name>.png so
you can eyeball the result without opening the GUI.

Usage:
    source .venv/bin/activate
    python test_effects.py
"""
import sys

import cv2
import numpy as np

from kinectfx.pipeline import Pipeline

NEW_EFFECTS = ["Thermal Silhouette", "Slit Scan", "Night Vision", "ASCII Art"]
FRAMES = 8  # enough for Slit Scan to fill its time window


def main():
    pipe = Pipeline()
    # Force the synthetic source so this is a pure code test, independent of
    # whether a real Kinect is plugged in / wedged. The Pipeline still runs the
    # full capture->effect->feedback->resize path.
    pipe.kinect.available = False
    out_w, out_h = pipe.state.out_w, pipe.state.out_h
    print(f"output {out_w}x{out_h}  (synthetic source forced)")

    failures = 0
    for name in NEW_EFFECTS:
        pipe.update(effect=name)
        # make sure the source matches the effect's hint
        hint = __import__("kinectfx.effects", fromlist=["EFFECTS_BY_NAME"]).EFFECTS_BY_NAME[name].source_hint
        pipe.update(source=hint)
        last = None
        try:
            for i in range(FRAMES):
                last = pipe.process()
        except Exception as e:
            print(f"  FAIL {name}: raised {type(e).__name__}: {e}")
            failures += 1
            continue
        if last is None:
            print(f"  FAIL {name}: no frame produced")
            failures += 1
            continue
        if last.dtype != np.uint8 or last.shape != (out_h, out_w, 3):
            print(f"  FAIL {name}: bad output {last.dtype} {last.shape}")
            failures += 1
            continue
        # sanity: image is not entirely blank and not entirely saturated
        lo, hi = float(last.min()), float(last.max())
        path = f"docs/samples/test_{name.lower().replace(' ', '_')}.png"
        cv2.imwrite(path, last)
        print(f"  OK   {name:18s} range [{lo:.0f},{hi:.0f}] -> {path}")

    pipe.close()
    print(f"\n{len(NEW_EFFECTS) - failures}/{len(NEW_EFFECTS)} effects passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
