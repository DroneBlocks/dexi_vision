#!/usr/bin/env python3
"""
Draw the pictures for docs/bar-counter.md from one camera frame, using the
bar counter's own functions so the pictures show what the code really does.

    python3 docs/make_figures.py docs/images/frame.jpg docs/images
"""
import math
import os
import sys
import types

import cv2
import numpy as np

for m in ("rclpy", "rclpy.node", "dexi_interfaces", "dexi_interfaces.msg",
          "sensor_msgs", "sensor_msgs.msg"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["rclpy.node"].Node = object
sys.modules["dexi_interfaces.msg"].YoloDetection = None
sys.modules["dexi_interfaces.msg"].YoloDetectionArray = None
sys.modules["sensor_msgs.msg"].CompressedImage = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bar_counter_node import BarCounterNode  # noqa: E402

node = BarCounterNode.__new__(BarCounterNode)
node.__dict__.update(min_disc_area=800, core_gap=50, min_ink=0.02,
                     min_bar_frac=0.004, bar_aspect=1.2, min_circularity=0.70,
                     max_circularity=1.05, max_bar_width_frac=0.22,
                     bar_width_ratio=0.23, ink_tolerance=0.55)

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
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    save = lambda name, img: cv2.imwrite(os.path.join(outdir, name), img)

    save("0_camera.png", label(bgr, "What DEXI sees"))

    # 1a. dark pixels
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((gray < thr).astype(np.uint8)) * 255
    save("1a_dark.png", label(cv2.cvtColor(255 - dark, cv2.COLOR_GRAY2BGR),
                              "Step 1a: dark pixels are black"))

    # 1b. every dark blob, judged the way find_disc() judges it
    n, labels, comps, _ = cv2.connectedComponentsWithStats(dark)
    view = bgr.copy()
    kept = []
    for i in range(1, n):
        x0, y0, bw0, bh0, area = comps[i]
        if area < node.min_disc_area:
            continue
        pad = max(bw0, bh0) // 2 + 2
        xa, ya = max(0, x0 - pad), max(0, y0 - pad)
        xb, yb = min(W, x0 + bw0 + pad), min(H, y0 + bh0 + pad)
        g = gray[ya:yb, xa:xb]
        b, gr, rd = cv2.split(bgr[ya:yb, xa:xb])
        v = cv2.GaussianBlur(cv2.max(cv2.max(b, gr), rd), (5, 5), 0)
        blob = node.black_core(g, labels[ya:yb, xa:xb] == i)
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros(g.shape, np.uint8)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        m, lab, stats, cent = cv2.connectedComponentsWithStats(filled)
        for j in range(1, m):
            _x, _y, w, h, a = stats[j]
            if a < node.min_disc_area:
                continue
            r = max(w, h) / 2.0
            circ = a / (math.pi * r * r)
            cx, cy = cent[j]
            round_ = 0.7 < w / max(1, h) < 1.45 and node.min_circularity <= circ <= node.max_circularity
            ok = round_ and node.stands_out(v, lab == j, cx, cy, r)
            mask = np.zeros((H, W), np.uint8)
            mask[ya:yb, xa:xb] = (lab == j).astype(np.uint8) * 255
            c2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, c2, -1, GREEN if ok else RED, 3)
            if ok:
                kept.append((cx + xa, cy + ya, r))
    save("1b_round.png", label(view, "Step 1b: green = round black disc, red = not"))

    # 1c. nearest the middle wins
    view = bgr.copy()
    cv2.drawMarker(view, (W // 2, H // 2), YELLOW, cv2.MARKER_CROSS, 40, 3)
    best = min(kept, key=lambda k: math.hypot(k[0] - W / 2, k[1] - H / 2))
    for cx, cy, r in kept:
        win = (cx, cy, r) == best
        cv2.circle(view, (int(cx), int(cy)), int(r), GREEN if win else (160, 160, 160), 3)
        cv2.line(view, (W // 2, H // 2), (int(cx), int(cy)), GREEN if win else (160, 160, 160), 2)
    save("1c_nearest.png", label(view, "Step 1c: the disc nearest the middle wins"))

    # 2-3. bars inside the chosen disc, zoomed in
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), int(r * 0.88), 255, -1)
    inside = gray[mask > 0]
    t2, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2 = max(t2, float(np.percentile(inside, 10)) + node.core_gap)
    bars = ((gray > t2) & (mask > 0)).astype(np.uint8)

    s = int(r * 1.5)
    x0, y0 = max(0, cx - s), max(0, cy - s)
    zoom = lambda img: cv2.resize(img[y0:cy + s, x0:cx + s], (400, 400),
                                  interpolation=cv2.INTER_NEAREST)
    step2 = bgr.copy()
    step2[bars > 0] = CYAN
    cv2.circle(step2, (cx, cy), int(r * 0.88), YELLOW, 1)
    save("2_bar_pixels.png", label(zoom(step2), "Step 2: bright pixels inside"))

    step3 = bgr.copy()
    cnts, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([c for c in cnts if cv2.contourArea(c) >= node.min_bar_frac * (mask > 0).sum()],
                  key=lambda c: cv2.boundingRect(c)[0])
    step3 = zoom(step3)
    k = 400 / (2 * s)
    for i, c in enumerate(cnts, 1):
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = ((box - [x0, y0]) * k).astype(np.int32)
        cv2.polylines(step3, [box], True, GREEN, 2)
        top = box[np.argmin(box[:, 1])]
        cv2.putText(step3, str(i), (int(top[0]) - 8, int(top[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    save("3_count.png", label(step3, f"Step 3: a box around each bar = {len(cnts)}"))

    # 5. the vote, drawn
    votes = [f"bars_{len(cnts)}"] * 2 + ["?"] + [f"bars_{len(cnts)}"] * 2
    strip = np.full((120, 640, 3), 255, np.uint8)
    for i, vtx in enumerate(votes):
        x = 15 + i * 125
        cv2.rectangle(strip, (x, 40), (x + 110, 100), GREEN if vtx != "?" else (180, 180, 180), -1)
        cv2.putText(strip, vtx, (x + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(strip, "Last 5 frames: 4 agree, so the answer is sent", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    save("5_vote.png", strip)

    print(node.classify(bgr))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
