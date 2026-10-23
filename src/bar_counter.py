"""
Count the white bars inside a black disc.

The computer vision behind bar_counter_node.py, with no ROS imports, so the
tests and the doc figures run the same code the node does.

    reading = BarCounter().read(bgr)
    reading.label        # bars_1, bars_2, bars_3, bars_0, or None when unsure

Plain OpenCV, no trained model. The disc images differ only in a few thin
white bars, which a detector like YOLO tends to confuse; counting the bars
directly is deterministic.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

BLUR = (5, 5)
# Width / height of a round disc's bounding box.
SQUARE_BBOX = (0.7, 1.45)
# Ring sampled around a disc to judge contrast, in disc radii.
RING = (1.05, 1.4)
# Bars are looked for inside this fraction of the disc radius, clear of the rim.
INNER = 0.88


@dataclass
class Params:
    """Tuning, named as in config/bar_counter_params.yaml."""
    min_disc_area: int = 800
    min_circularity: float = 0.70
    max_circularity: float = 1.05
    core_gap: int = 50
    min_ink_fraction: float = 0.02
    min_bar_area_fraction: float = 0.004
    bar_aspect_min: float = 1.2
    max_bar_width_frac: float = 0.22
    bar_width_ratio: float = 0.23
    ink_tolerance: float = 0.55


@dataclass
class Disc:
    cx: float
    cy: float
    r: float
    contour: np.ndarray     # frame coordinates
    ok: bool                # round, and darker than its surround


@dataclass
class Reading:
    label: str | None    # None means "not sure", not "no bars"
    confidence: float
    bbox: tuple[float, float, float, float] | None   # normalized x1, y1, x2, y2
    note: str


def prepare(bgr):
    """Blurred gray and value (brightest channel) images of a frame."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), BLUR, 0)
    b, g, r = cv2.split(bgr)
    value = cv2.GaussianBlur(cv2.max(cv2.max(b, g), r), BLUR, 0)
    return gray, value


class BarCounter:
    def __init__(self, params: Params = None):
        self.p = params or Params()

    # -- step 1: the disc ------------------------------------------------
    def dark(self, gray):
        """Pixels darker than the frame's Otsu threshold."""
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (gray < thr).astype(np.uint8)

    def discs(self, gray, value):
        """Every dark shape big enough to be a disc, each judged round or not."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(self.dark(gray))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.p.min_disc_area:
                continue
            # Work inside the blob's box: whole-frame masks per blob are too
            # slow for a CM4 in a cluttered scene.
            core = self.black_core(gray[y:y + h, x:x + w], labels[y:y + h, x:x + w] == i)
            # Outer contours only, which fills the holes the bars cut.
            contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y)))
            for c in contours:
                disc = self.judge(c, value)
                if disc is not None:
                    yield disc

    def black_core(self, gray, blob):
        """
        Keep only the black part of a dark blob.

        The frame-wide threshold can join a gray surround (the inside wall of
        a cup) to the black disc. A second Otsu over the blob's own pixels
        splits black from gray. It is only applied when the two halves are
        far apart, so a disc of one tone is never cut in two.
        """
        vals = gray[blob]
        t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = vals[vals <= t], vals[vals > t]
        if lo.size and hi.size and hi.mean() - lo.mean() >= self.p.core_gap:
            blob = blob & (gray <= t)
        return blob.astype(np.uint8)

    def judge(self, contour, value):
        """A Disc for a filled contour, or None when it is too small."""
        bx, by, w, h = cv2.boundingRect(contour)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, -1, offset=(-bx, -by))
        m = cv2.moments(filled, binaryImage=True)
        if m["m00"] < self.p.min_disc_area:
            return None
        cx, cy = bx + m["m10"] / m["m00"], by + m["m01"] / m["m00"]
        r = max(w, h) / 2.0
        # Area over the area of a circle as wide: 1.0 for a circle, 4/pi for
        # a filled square, so the upper bound rejects dark rectangles such as
        # a floor region.
        circularity = m["m00"] / (math.pi * r * r)
        ok = (SQUARE_BBOX[0] < w / h < SQUARE_BBOX[1]
              and self.p.min_circularity <= circularity <= self.p.max_circularity
              and self.stands_out(value, value[by:by + h, bx:bx + w][filled > 0], cx, cy, r))
        return Disc(cx, cy, r, contour, ok)

    def stands_out(self, value, inside, cx, cy, r):
        """
        True if the disc is clearly darker than a ring around it.

        A printed disc is black on a lighter surface; a soft shadow is not.
        Judged on value rather than gray, so a saturated color such as red,
        which is dark in gray, does not pass.
        """
        R = int(r * RING[1])
        x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
        window = value[y0:int(cy) + R + 1, x0:int(cx) + R + 1]
        ring = np.zeros(window.shape, np.uint8)
        center = (int(cx) - x0, int(cy) - y0)
        cv2.circle(ring, center, R, 1, -1)
        cv2.circle(ring, center, int(r * RING[0]), 0, -1)
        around = window[ring > 0]
        return around.size > 0 and np.median(around) - np.median(inside) >= self.p.core_gap

    def nearest_disc(self, gray, value):
        """
        The round disc nearest the frame center, or None.

        Not the largest dark shape: clutter at the frame edge can be bigger.
        Not the most circular either: the neighboring cups are often in
        frame, and the one under the camera is the one nearest the middle.
        """
        H, W = gray.shape
        ok = [d for d in self.discs(gray, value) if d.ok]
        return min(ok, key=lambda d: math.hypot(d.cx - W / 2, d.cy - H / 2), default=None)

    # -- steps 2 to 4: the bars --------------------------------------------
    def bar_pixels(self, gray, cx, cy, r):
        """The bright pixels inside the disc, and how many pixels were searched."""
        inner = np.zeros(gray.shape, np.uint8)
        cv2.circle(inner, (cx, cy), int(r * INNER), 1, -1)
        inside = gray[inner > 0]
        t, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # On a blank disc Otsu splits camera noise, so a bar pixel must also be
        # well above the disc's own black level.
        t = max(t, float(np.percentile(inside, 10)) + self.p.core_gap)
        return ((gray > t) & (inner > 0)).astype(np.uint8), inside.size

    def bar_shapes(self, bars, searched):
        """minAreaRect of every bright shape big enough to be part of a bar."""
        contours, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.minAreaRect(c) for c in contours
                if cv2.contourArea(c) >= self.p.min_bar_area_fraction * searched]

    def read(self, bgr):
        H, W = bgr.shape[:2]
        gray, value = prepare(bgr)
        disc = self.nearest_disc(gray, value)
        if disc is None:
            return Reading(None, 0.0, None, "no disc")
        cx, cy, r = int(disc.cx), int(disc.cy), int(disc.r)
        bbox = (max(0.0, (cx - r) / W), max(0.0, (cy - r) / H),
                min(1.0, (cx + r) / W), min(1.0, (cy + r) / H))

        bars, searched = self.bar_pixels(gray, cx, cy, r)
        if searched < 50:
            return Reading(None, 0.0, None, "disc too small")
        ink = bars.sum() / searched
        if ink < self.p.min_ink_fraction:
            return Reading("bars_0", 0.90, bbox, f"ink {ink:.1%}, blank disc")

        # The bars' angle depends on the drone's heading, so each shape is
        # measured along its own axes with minAreaRect.
        widths, lengths = [], []
        max_width = 2 * r * self.p.max_bar_width_frac
        for _center, (d1, d2), _angle in self.bar_shapes(bars, searched):
            width, length = min(d1, d2), max(d1, d2)
            if length < width * self.p.bar_aspect_min:
                continue
            if width > max_width:
                # Wider than any single bar: the bars have merged, usually
                # from tilt. Abstain rather than report one wide bar.
                return Reading(None, 0.0, bbox, f"shape {width:.0f}px wide, bars merged")
            widths.append(round(width))
            lengths.append(length)

        k = len(widths)
        if k not in (1, 2, 3):
            return Reading(None, 0.0, bbox, f"{k} bar shapes, ink {ink:.0%}")

        # Total bar area grows with the bar count however the shapes split, so
        # it catches merges and splits the shape count misses. One bar's area
        # comes from the longest bar (printed bars are 0.23 as wide as long),
        # not from the disc, whose visible size shrinks when a cup wall hides
        # its rim.
        longest = max(lengths)
        k_area = bars.sum() / (self.p.bar_width_ratio * longest * longest)
        if abs(k_area - k) > self.p.ink_tolerance:
            return Reading(None, 0.0, bbox, f"{k} shapes but area says {k_area:.1f} bars")

        # Printed bars are identical, so uneven widths lower the confidence.
        conf = 0.85
        if k > 1:
            spread = float(np.std(widths)) / max(1.0, float(np.mean(widths)))
            conf = max(0.40, min(0.95, 0.95 - spread))
        return Reading(f"bars_{k}", conf, bbox, f"{k} bars, widths {sorted(widths)}, r {r}px")
