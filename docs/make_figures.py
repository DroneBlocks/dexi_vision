#!/usr/bin/env python3
"""
Draw the pictures for docs/bar-counter.md from one camera frame, using the
bar counter's own functions so the pictures show what the code really does.

    python3 docs/make_figures.py docs/images/frame.jpg docs/images
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bar_counter import INNER, BarCounter, prepare  # noqa: E402

counter = BarCounter()

GREEN, RED, YELLOW, CYAN = (60, 200, 60), (50, 50, 230), (0, 220, 255), (230, 200, 0)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    bgr = cv2.imread(src)
    H, W = bgr.shape[:2]
    gray, value = prepare(bgr)
    save = lambda name, img: cv2.imwrite(os.path.join(outdir, name), img)

    save("0_camera.png", label(bgr, "What DEXI sees"))

    # 1a. dark pixels
    dark = counter.dark(gray) * 255
    save("1a_dark.png", label(cv2.cvtColor(255 - dark, cv2.COLOR_GRAY2BGR),
                              "Step 1a: dark pixels are black"))

    # 1b. every dark shape, judged by the counter
    view = bgr.copy()
    discs = list(counter.discs(gray, value))
    for d in discs:
        cv2.drawContours(view, [d.contour], -1, GREEN if d.ok else RED, 3)
    save("1b_round.png", label(view, "Step 1b: green = round black disc, red = not"))

    # 1c. nearest the middle wins
    view = bgr.copy()
    cv2.drawMarker(view, (W // 2, H // 2), YELLOW, cv2.MARKER_CROSS, 40, 3)
    best = counter.nearest_disc(gray, value)
    for d in discs:
        if not d.ok:
            continue
        color = GREEN if (d.cx, d.cy) == (best.cx, best.cy) else (160, 160, 160)
        cv2.circle(view, (int(d.cx), int(d.cy)), int(d.r), color, 3)
        cv2.line(view, (W // 2, H // 2), (int(d.cx), int(d.cy)), color, 2)
    save("1c_nearest.png", label(view, "Step 1c: the disc nearest the middle wins"))

    # 2-3. bars inside the chosen disc, zoomed in
    cx, cy, r = int(best.cx), int(best.cy), int(best.r)
    bars, searched = counter.bar_pixels(gray, cx, cy, r)

    s = int(r * 1.5)
    x0, y0 = max(0, cx - s), max(0, cy - s)
    zoom = lambda img: cv2.resize(img[y0:cy + s, x0:cx + s], (400, 400),
                                  interpolation=cv2.INTER_NEAREST)
    step2 = bgr.copy()
    step2[bars > 0] = CYAN
    cv2.circle(step2, (cx, cy), int(r * INNER), YELLOW, 1)
    save("2_bar_pixels.png", label(zoom(step2), "Step 2: bright pixels inside"))

    step3 = bgr.copy()
    rects = sorted(counter.bar_shapes(bars, searched), key=lambda rect: rect[0][0])
    step3 = zoom(step3)
    k = 400 / (2 * s)
    for i, rect in enumerate(rects, 1):
        box = cv2.boxPoints(rect)
        box = ((box - [x0, y0]) * k).astype(np.int32)
        cv2.polylines(step3, [box], True, GREEN, 2)
        top = box[np.argmin(box[:, 1])]
        cv2.putText(step3, str(i), (int(top[0]) - 8, int(top[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    save("3_count.png", label(step3, f"Step 3: a box around each bar = {len(rects)}"))

    # 5. the vote, drawn
    votes = [f"bars_{len(rects)}"] * 2 + ["?"] + [f"bars_{len(rects)}"] * 2
    strip = np.full((120, 640, 3), 255, np.uint8)
    for i, vtx in enumerate(votes):
        x = 15 + i * 125
        cv2.rectangle(strip, (x, 40), (x + 110, 100), GREEN if vtx != "?" else (180, 180, 180), -1)
        cv2.putText(strip, vtx, (x + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(strip, "Last 5 frames: 4 agree, so the answer is sent", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    save("5_vote.png", strip)

    print(counter.read(bgr))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
