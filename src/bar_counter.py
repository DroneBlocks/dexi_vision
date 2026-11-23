"""
Count the white bars on a bridge cup sticker.

The computer vision behind bar_counter_node.py, with no ROS imports, so the
tests and the doc figures run the same code the node does.

    reading = BarCounter().read(bgr)
    reading.label        # bars_1, bars_2, bars_3, bars_0, or None when unsure

Two ways in, run on every frame:

- disc first: find a black disc that stands out from a lighter surface, then
  count the bars inside it. Works on printed stickers on a bench or floor.
- bars first: find groups of bright, thin, parallel bars on dark ground. On
  the real field the sticker sits in a clear cup on a black bridge, so the
  disc is the same black as the bridge and cannot be found by darkness.

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
# Percentile of the ring taken as "the surface around the disc".
RING_LIGHT = 90
# Bars are looked for inside this fraction of the disc radius, clear of the rim.
INNER = 0.88
# Bars first: neighborhood (px) a bar must outshine, and the shape limits for
# one bar. The block is wider than a bar seen up close.
BAR_BLOCK = 61
BAR_MIN_LENGTH = 12
BAR_MIN_ASPECT = 2.5
BAR_MIN_FILL = 0.7
# Bars of one sticker: parallel within this angle, similar in length and
# width, side by side rather than end to end.
GROUP_MAX_ANGLE = math.radians(15)
GROUP_MIN_LENGTH_RATIO = 0.65
GROUP_MIN_WIDTH_RATIO = 0.5
GROUP_MAX_ALONG = 0.25
GROUP_ACROSS = (1.5, 0.85)   # spacing: at least 1.5 widths, at most 0.85 lengths
# Ground sampled around a bar group, in bar lengths past the group's edge.
GROUND_RING = 0.2
GROUND_DARK = 0.9            # fraction of that ring clearly darker than the bars


@dataclass
class Params:
    """Tuning, named as in config/bar_counter_params.yaml."""
    min_disc_area: int = 800
    min_circularity: float = 0.70
    max_circularity: float = 1.05
    min_solidity: float = 0.9
    core_gap: int = 50
    min_ink_fraction: float = 0.02
    min_bar_area_fraction: float = 0.004
    bar_aspect_min: float = 1.2
    min_bar_length_frac: float = 0.3
    max_bar_width_frac: float = 0.22
    ink_tolerance: float = 0.55
    bar_contrast: int = 40
    min_bar_level: int = 170
    max_center_dist: float = 0.7
    max_stray: float = 0.3
    max_ring_std: float = 16.0


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


@dataclass
class Bar:
    x: float
    y: float
    length: float
    width: float
    angle: float            # radians, along the bar
    contour: np.ndarray


class BarCounter:
    def __init__(self, params: Params = None):
        self.p = params or Params()

    def read(self, bgr):
        """
        Read the cup nearest the frame center.

        Both ways run. When they see the same cup and disagree, abstain. When
        they see different cups, the one nearer the center is the cup under
        the drone.
        """
        H, W = bgr.shape[:2]
        a = self.off_center(self.read_disc_first(bgr), W, H)
        b = self.read_bars_first(bgr)
        if a.label is None:
            return b
        if b.label is None:
            return a
        (ax, ay), (bx, by) = self.center(a, W, H), self.center(b, W, H)
        if math.hypot(ax - bx, ay - by) < (b.bbox[2] - b.bbox[0]) * W / 2:
            if a.label == b.label:
                return a
            return Reading(None, 0.0, a.bbox, f"disc says {a.label}, bars say {b.label}")
        da = math.hypot(ax - W / 2, ay - H / 2)
        db = math.hypot(bx - W / 2, by - H / 2)
        return a if da < db else b

    @staticmethod
    def center(reading, W, H):
        x1, y1, x2, y2 = reading.bbox
        return (x1 + x2) / 2 * W, (y1 + y2) / 2 * H

    def off_center(self, reading, W, H):
        """
        Drop a reading far from the frame center. The drone hovers over the
        cup it is scoring; a cup near the edge is a neighbor.
        """
        if reading.bbox is None:
            return reading
        x, y = self.center(reading, W, H)
        if math.hypot(x - W / 2, y - H / 2) > self.p.max_center_dist * H / 2:
            return Reading(None, 0.0, None, "disc off center")
        return reading

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
        # A shape cut off by the frame edge is not a whole disc. This also
        # drops a dark floor or mat that fills the frame, which can otherwise
        # pass as a round disc nearest the center.
        H, W = value.shape
        inside_frame = bx > 0 and by > 0 and bx + w < W and by + h < H
        # A printed disc is convex, even where a cup wall hides part of it;
        # a ragged patch of shadow or grass is not.
        solidity = m["m00"] / max(1.0, cv2.contourArea(cv2.convexHull(contour)))
        ok = (inside_frame
              and solidity >= self.p.min_solidity
              and SQUARE_BBOX[0] < w / h < SQUARE_BBOX[1]
              and self.p.min_circularity <= circularity <= self.p.max_circularity
              and self.stands_out(value, value[by:by + h, bx:bx + w][filled > 0], cx, cy, r))
        return Disc(cx, cy, r, contour, ok)

    def stands_out(self, value, inside, cx, cy, r):
        """
        True if the disc is clearly darker than the light part of a ring
        around it.

        A printed disc is black on a lighter surface; a soft shadow is not.
        The ring's 90th percentile is used, not its median: a sticker cut out
        of white paper and set on a dark mat or floor has only a thin light
        rim, and the dark ground beyond it fills most of the ring. Judged on
        value rather than gray, so a saturated color such as red, which is
        dark in gray, does not pass.
        """
        R = int(r * RING[1])
        x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
        window = value[y0:int(cy) + R + 1, x0:int(cx) + R + 1]
        ring = np.zeros(window.shape, np.uint8)
        center = (int(cx) - x0, int(cy) - y0)
        cv2.circle(ring, center, R, 1, -1)
        cv2.circle(ring, center, int(r * RING[0]), 0, -1)
        around = window[ring > 0]
        return around.size > 0 and np.percentile(around, RING_LIGHT) - np.median(inside) >= self.p.core_gap

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
    def bar_pixels(self, gray, cx, cy, r, contour=None):
        """The bright pixels inside the disc, and how many pixels were searched."""
        inner = np.zeros(gray.shape, np.uint8)
        cv2.circle(inner, (cx, cy), int(r * INNER), 1, -1)
        if contour is not None:
            # Only inside the disc's outline, pulled in from its edge. When a
            # cup wall hides part of the disc, the circle alone reaches past the
            # black and picks up slivers of the bright rim beyond it. The convex
            # hull, not the outline itself: a bar that runs out to the hidden
            # edge cuts a notch in the outline, and the hull keeps that bar.
            disc = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(disc, [cv2.convexHull(contour)], -1, 1, -1)
            e = max(1, int(r * 0.06))
            inner &= cv2.erode(disc, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
            if not inner.any():
                return inner, 0
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

    def read_disc_first(self, bgr):
        H, W = bgr.shape[:2]
        gray, value = prepare(bgr)
        disc = self.nearest_disc(gray, value)
        if disc is None:
            return Reading(None, 0.0, None, "no disc")
        cx, cy, r = int(disc.cx), int(disc.cy), int(disc.r)
        bbox = (max(0.0, (cx - r) / W), max(0.0, (cy - r) / H),
                min(1.0, (cx + r) / W), min(1.0, (cy + r) / H))

        bars, searched = self.bar_pixels(gray, cx, cy, r, disc.contour)
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
            if length < width * self.p.bar_aspect_min or length < 2 * r * self.p.min_bar_length_frac:
                continue
            if width > max_width:
                # Wider than any single bar: the bars have merged, usually
                # from tilt. Abstain rather than report one wide bar.
                return Reading(None, 0.0, bbox, f"shape {width:.0f}px wide, bars merged")
            widths.append(width)
            lengths.append(length)

        k = len(widths)
        if k not in (1, 2, 3):
            return Reading(None, 0.0, bbox, f"{k} bar shapes, ink {ink:.0%}")

        # Total bar area grows with the bar count however the shapes split, so
        # it catches merges and splits the shape count misses. One bar's area
        # is the median bar width times the longest bar, both measured in this
        # frame, so wide and thin bar artwork both work. The disc is not used:
        # its visible size shrinks when a cup wall hides its rim.
        one_bar = float(np.median(widths)) * max(lengths)
        k_area = bars.sum() / max(1.0, one_bar)
        if abs(k_area - k) > self.p.ink_tolerance:
            return Reading(None, 0.0, bbox, f"{k} shapes but area says {k_area:.1f} bars")

        # Printed bars are identical, so uneven widths lower the confidence.
        conf = 0.85
        if k > 1:
            spread = float(np.std(widths)) / max(1.0, float(np.mean(widths)))
            conf = max(0.40, min(0.95, 0.95 - spread))
        return Reading(f"bars_{k}", conf, bbox, f"{k} bars, widths {sorted(round(w) for w in widths)}, r {r}px")

    # -- bars first ---------------------------------------------------------
    def bright_bars(self, gray):
        """
        Every bright, thin, filled shape that could be one bar, and the mask
        of bright pixels it came from.

        Bright means brighter than its own neighborhood by bar_contrast, not
        brighter than a frame-wide cut-off, so a bar shows up on a black
        bridge, a gray floor or in shade alike.
        """
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        bright = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       BAR_BLOCK, -self.p.bar_contrast)
        # A bar can sit inside a bright shape's hole, as on a black sticker in
        # a lit clear cup, so take every outer boundary, nested ones too.
        contours, hierarchy = cv2.findContours(bright, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape
        bars = []
        for c, h in zip(contours, hierarchy[0] if hierarchy is not None else []):
            if h[3] != -1:
                continue
            area = cv2.contourArea(c)
            if area < 20:
                continue
            (x, y), (d1, d2), angle = cv2.minAreaRect(c)
            length, width = max(d1, d2), max(1.0, min(d1, d2))
            if (length < BAR_MIN_LENGTH or length / width < BAR_MIN_ASPECT
                    or length > 0.5 * min(H, W) or area / (d1 * d2 + 1e-6) < BAR_MIN_FILL):
                continue
            theta = math.radians(angle if d1 >= d2 else angle + 90)
            bars.append(Bar(x, y, length, width, theta, c))
        return bars, bright

    @staticmethod
    def same_sticker(a, b):
        """True if two bars sit side by side the way a sticker's bars do."""
        turn = abs((a.angle - b.angle + math.pi / 2) % math.pi - math.pi / 2)
        if turn > GROUP_MAX_ANGLE:
            return False
        if min(a.length, b.length) / max(a.length, b.length) < GROUP_MIN_LENGTH_RATIO:
            return False
        if min(a.width, b.width) / max(a.width, b.width) < GROUP_MIN_WIDTH_RATIO:
            return False
        length, width = (a.length + b.length) / 2, (a.width + b.width) / 2
        ux, uy = math.cos(a.angle), math.sin(a.angle)
        dx, dy = b.x - a.x, b.y - a.y
        along, across = abs(dx * ux + dy * uy), abs(-dx * uy + dy * ux)
        return along <= GROUP_MAX_ALONG * length and \
            GROUP_ACROSS[0] * width <= across <= GROUP_ACROSS[1] * length

    def bar_groups(self, bars):
        """Bars joined into stickers: connected groups of same_sticker pairs."""
        links = {i: [j for j in range(len(bars)) if j != i and self.same_sticker(bars[i], bars[j])]
                 for i in range(len(bars))}
        seen, groups = set(), []
        for i in links:
            if i in seen:
                continue
            stack, group = [i], []
            seen.add(i)
            while stack:
                k = stack.pop()
                group.append(bars[k])
                for j in links[k]:
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
            groups.append(group)
        return groups

    def on_dark_ground(self, gray, bright, group):
        """
        Where a group sits and how long its bars are, or None if it is not
        a sticker's bars on a dark disc.

        - A ring just outside the group, still on the disc, is clearly darker
          than the bars.
        - Inside that ring there is no other bright shape. A bar the camera
          blurred into a blob would be one, so the count would be short:
          abstain instead. So would a white airplane's wings.
        - A lone bar also needs smooth ground around it. Dirt and grass are
          textured; a printed disc is not.
        """
        cx = float(np.mean([b.x for b in group]))
        cy = float(np.mean([b.y for b in group]))
        length = float(np.mean([b.length for b in group]))
        width = float(np.mean([b.width for b in group]))
        spread = max(math.hypot(b.x - cx, b.y - cy) for b in group) + width
        r_in = max(0.55 * length, spread + width)
        r_out = r_in + GROUND_RING * length
        H, W = gray.shape
        if cx - r_out < 0 or cy - r_out < 0 or cx + r_out >= W or cy + r_out >= H:
            return None
        center = (int(cx), int(cy))
        ring = np.zeros(gray.shape, np.uint8)
        cv2.circle(ring, center, int(r_out), 1, -1)
        cv2.circle(ring, center, int(r_in), 0, -1)
        on_bars = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(on_bars, [b.contour for b in group], -1, 1, -1)
        bar_level = float(np.median(gray[on_bars > 0]))
        ground = gray[ring > 0]
        if bar_level < self.p.min_bar_level or bar_level - np.median(ground) < 60:
            return None
        if np.mean(ground < bar_level - self.p.core_gap) < GROUND_DARK:
            return None
        inner = np.zeros(gray.shape, np.uint8)
        cv2.circle(inner, center, int(r_in), 1, -1)
        near_bars = cv2.dilate(on_bars, np.ones((5, 5), np.uint8))
        stray = float(((bright > 0) & (inner > 0) & (near_bars == 0)).sum())
        if stray > self.p.max_stray * np.mean([cv2.contourArea(b.contour) for b in group]):
            return None
        if len(group) == 1:
            close = np.zeros(gray.shape, np.uint8)
            cv2.circle(close, center, int(0.9 * length), 1, -1)
            cv2.circle(close, center, int(0.6 * length), 0, -1)
            if np.std(gray[close > 0]) > self.p.max_ring_std:
                return None
        return cx, cy, length

    def read_bars_first(self, bgr):
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bars, bright = self.bright_bars(gray)
        best = None
        for group in self.bar_groups(bars):
            if len(group) > 3:
                continue
            found = self.on_dark_ground(gray, bright, group)
            if found is None:
                continue
            cx, cy, length = found
            dist = math.hypot(cx - W / 2, cy - H / 2)
            if dist > self.p.max_center_dist * H / 2:
                continue
            if best is None or dist < best[0]:
                best = (dist, group, cx, cy, length)
        if best is None:
            return Reading(None, 0.0, None, "no bars on dark ground")
        _, group, cx, cy, r = best
        k = len(group)
        bbox = (max(0.0, (cx - r) / W), max(0.0, (cy - r) / H),
                min(1.0, (cx + r) / W), min(1.0, (cy + r) / H))
        return Reading(f"bars_{k}", 0.80, bbox, f"{k} bars on dark ground, length {r:.0f}px")
