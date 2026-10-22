#!/usr/bin/env python3
"""
Bar counter.

Counts the white bars inside a black disc and publishes the count on
/bar_detections as a YoloDetectionArray, so it reads like any other detector:

    1 bar  -> bars_1
    2 bars -> bars_2
    3 bars -> bars_3
    0 bars -> bars_0   (plain black disc)

Nothing is published when the disc cannot be read (too far, tilted, bars
merged). No message means "not sure", not "no bars".

The node knows nothing about the game. On the AVR 2026 field the discs are the
bridge-cup stickers, and the Node-RED flow maps bars_1/2/3 to red/green/blue.

Plain OpenCV, no trained model. The three disc images differ only in a few
thin white bars, which a detector like YOLO tends to confuse; counting the
bars directly is deterministic.

Same shape as color_detection_node.py in this package: a params file, a
detection_frequency gate and compressed-image input. Publishes on its own
topic so it can run alongside the YOLO node.
"""
import math
from collections import Counter, deque

import cv2
import numpy as np
import rclpy
from dexi_interfaces.msg import YoloDetection, YoloDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class BarCounterNode(Node):
    def __init__(self):
        super().__init__("bar_counter_node")

        self.declare_parameter("detection_frequency", 2.0)
        self.declare_parameter("min_disc_area", 800)
        self.declare_parameter("min_ink_fraction", 0.02)
        self.declare_parameter("min_bar_area_fraction", 0.004)
        self.declare_parameter("bar_aspect_min", 1.2)
        self.declare_parameter("min_circularity", 0.70)
        self.declare_parameter("max_circularity", 1.05)
        self.declare_parameter("max_bar_width_frac", 0.22)
        self.declare_parameter("bar_width_ratio", 0.23)
        self.declare_parameter("ink_tolerance", 0.55)
        self.declare_parameter("core_gap", 50)
        self.declare_parameter("vote_window", 5)
        self.declare_parameter("vote_min", 3)
        self.declare_parameter("verbose", False)

        self.freq = self.get_parameter("detection_frequency").value
        self.min_disc_area = self.get_parameter("min_disc_area").value
        self.min_ink = self.get_parameter("min_ink_fraction").value
        self.min_bar_frac = self.get_parameter("min_bar_area_fraction").value
        self.bar_aspect = self.get_parameter("bar_aspect_min").value
        self.min_circularity = self.get_parameter("min_circularity").value
        self.max_circularity = self.get_parameter("max_circularity").value
        self.max_bar_width_frac = self.get_parameter("max_bar_width_frac").value
        self.bar_width_ratio = self.get_parameter("bar_width_ratio").value
        self.ink_tolerance = self.get_parameter("ink_tolerance").value
        self.core_gap = self.get_parameter("core_gap").value
        self.vote_min = self.get_parameter("vote_min").value
        self.votes = deque(maxlen=self.get_parameter("vote_window").value)
        self.verbose = self.get_parameter("verbose").value

        self.min_interval = 1.0 / max(0.1, self.freq)
        self.last_t = 0.0

        self.pub = self.create_publisher(YoloDetectionArray, "/bar_detections", 10)
        self.create_subscription(CompressedImage, "/cam0/image_raw/compressed",
                                 self.cb, 10)

        self.get_logger().info("bar_counter_node up")
        self.get_logger().info("  publishing : /bar_detections")
        self.get_logger().info(f"  frequency  : {self.freq} Hz")
        self.get_logger().info("  classes    : bars_1, bars_2, bars_3, bars_0")

    # ------------------------------------------------------------------
    def find_disc(self, gray, bgr):
        """
        Return the round black disc nearest the image center, or None.

        Not the largest dark blob: clutter at the frame edge can be bigger
        than the disc. Not the most circular either: neighboring discs are
        often in frame, and the one under the camera is the one nearest the
        middle. Circularity = area / (pi * r^2), 1.0 for a true circle.
        """
        thr, _ = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = ((gray < thr).astype(np.uint8)) * 255
        n, labels, comps, _ = cv2.connectedComponentsWithStats(dark)
        H, W = gray.shape
        best, best_dist = None, float("inf")
        for i in range(1, n):
            x0, y0, bw0, bh0, area = comps[i]
            if area < self.min_disc_area:
                continue
            # Work on a crop around the blob; full-frame masks per blob are
            # too slow for a CM4 in a cluttered scene. The pad leaves room for
            # the 1.4 r ring that stands_out() samples.
            pad = max(bw0, bh0) // 2 + 2
            xa, ya = max(0, x0 - pad), max(0, y0 - pad)
            xb, yb = min(W, x0 + bw0 + pad), min(H, y0 + bh0 + pad)
            g = gray[ya:yb, xa:xb]
            b, gr, rd = cv2.split(bgr[ya:yb, xa:xb])
            v = cv2.GaussianBlur(cv2.max(cv2.max(b, gr), rd), (5, 5), 0)
            blob = self.black_core(g, labels[ya:yb, xa:xb] == i)
            # Fill the holes the bars cut, so a disc with wide bars still
            # scores as round. Works at any scale, unlike a fixed-size close.
            cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            blob = np.zeros(g.shape, np.uint8)
            cv2.drawContours(blob, cnts, -1, 255, -1)
            m, lab, stats, cent = cv2.connectedComponentsWithStats(blob)
            for j in range(1, m):
                _x, _y, w, h, a = stats[j]
                if a < self.min_disc_area:
                    continue
                if not (0.7 < w / max(1, h) < 1.45):   # roughly square bbox
                    continue
                r = max(w, h) / 2.0
                circ = a / (math.pi * r * r)
                if not (self.min_circularity <= circ <= self.max_circularity):
                    # Bounded both ways: a circle scores 1.0 and a filled
                    # square 4/pi = 1.27, so the upper bound rejects large dark
                    # rectangles such as a floor region.
                    continue
                cx, cy = cent[j]
                if not self.stands_out(v, lab == j, cx, cy, r):
                    continue
                cx, cy = cx + xa, cy + ya
                dist = math.hypot(cx - W / 2.0, cy - H / 2.0)
                if dist < best_dist:
                    best_dist = dist
                    best = (w, h, int(cx), int(cy))
        return best

    def stands_out(self, value, disc, cx, cy, r):
        """
        True if the disc is clearly darker than a ring around it.

        A printed disc is black on a lighter surface; a soft shadow is not.
        Judged on value (the brightest channel) rather than grey, so a
        saturated color such as red, which is dark in grey, does not pass.
        """
        ring = np.zeros(value.shape, np.uint8)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.4), 255, -1)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.05), 0, -1)
        around = value[ring > 0]
        if around.size == 0:
            return False
        return np.median(around) - np.median(value[disc]) >= self.core_gap

    def black_core(self, gray, blob):
        """
        Keep only the black part of a dark blob.

        The image-wide threshold can join a grey surround (the inside wall of
        a cup) to the black disc. A second Otsu over the blob's own pixels
        splits black from grey. It is only applied when the two halves are
        far apart, so a plain disc of one tone is never cut in two.
        """
        vals = gray[blob]
        t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = vals[vals <= t], vals[vals > t]
        if lo.size == 0 or hi.size == 0 or hi.mean() - lo.mean() < self.core_gap:
            return blob
        return blob & (gray <= t)

    def classify(self, bgr):
        """-> (label, confidence, bbox_xyxy_normalised, note)"""
        H, W = bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        disc = self.find_disc(gray, bgr)
        if disc is None:
            return None, 0.0, None, "no disc"
        w, h, cx, cy = disc
        r = int(max(w, h) / 2)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.88), 255, -1)
        npx = int((mask > 0).sum())
        if npx < 50:
            return None, 0.0, None, "disc too small"

        inside = gray[mask > 0]
        t2, _ = cv2.threshold(inside, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # On a blank disc Otsu splits camera noise, so a bar pixel must also
        # be well above the disc's own black level.
        t2 = max(t2, float(np.percentile(inside, 10)) + self.core_gap)
        bars = ((gray > t2) & (mask > 0)).astype(np.uint8)
        ink = bars.sum() / float(npx)

        bbox = (max(0.0, (cx - r) / W), max(0.0, (cy - r) / H),
                min(1.0, (cx + r) / W), min(1.0, (cy + r) / H))

        if ink < self.min_ink:
            return "bars_0", 0.90, bbox, f"ink {ink*100:.1f}% -> blank disc"

        # The bars' angle in the image depends on the drone's heading, so each
        # blob is measured with minAreaRect, which gives its true short and
        # long axis at any angle.
        cnts, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        widths, lengths = [], []
        max_bar_w = 2 * r * self.max_bar_width_frac
        for c in cnts:
            a2 = cv2.contourArea(c)
            if a2 < self.min_bar_frac * npx:
                continue
            (_cxy, (d1, d2), _ang) = cv2.minAreaRect(c)
            bw, bh = min(d1, d2), max(d1, d2)          # short axis, long axis
            if bh < bw * self.bar_aspect:              # elongated at any angle
                continue
            if bw > max_bar_w:
                # Wider than any single bar (a bar is about 0.11 of the
                # diameter): the bars have merged, usually from tilt. Abstain
                # rather than report one wide bar.
                return None, 0.0, bbox, (f"blob {bw:.0f}px wide > {max_bar_w:.0f}px cap, "
                                         f"bars merged")
            widths.append(round(bw))
            lengths.append(bh)

        k = len(widths)
        if k not in (1, 2, 3):
            return None, 0.0, bbox, f"{k} bar-blobs (want 1-3) ink={ink*100:.0f}%"

        # Cross-check: total bar area is linear in bar count and does not
        # depend on how the blobs split, so it catches merges and splits the
        # blob count misses. One bar's area comes from the longest bar
        # (printed bars are 0.23 as wide as long) rather than from the disc,
        # whose visible size shrinks when a cup wall hides its rim.
        L = max(lengths)
        k_ink = bars.sum() / (self.bar_width_ratio * L * L)
        if abs(k_ink - k) > self.ink_tolerance:
            return None, 0.0, bbox, (f"{k} blobs but area implies {k_ink:.1f} bars")

        # Confidence from how uniform the bar widths are; printed bars are
        # identical, so a ragged set suggests noise.
        if k > 1:
            spread = float(np.std(widths)) / max(1.0, float(np.mean(widths)))
            conf = max(0.40, min(0.95, 0.95 - spread))
        else:
            conf = 0.85
        return (f"bars_{k}", conf, bbox,
                f"{k} bars w={sorted(widths)} r={r}px ink={ink*100:.0f}%")

    # ------------------------------------------------------------------
    def cb(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_t < self.min_interval:
            return
        self.last_t = now

        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return

        label, conf, bbox, note = self.classify(bgr)
        if self.verbose:
            self.get_logger().info(f"{label} ({conf:.2f}) {note}")

        # Publish only when one answer wins vote_min of the last vote_window
        # frames. Abstentions vote too, so a flickering read publishes nothing.
        self.votes.append((label, conf, bbox))
        top, count = Counter(v[0] for v in self.votes).most_common(1)[0]
        if top is None or count < self.vote_min:
            return
        agree = [v for v in self.votes if v[0] == top]
        label, bbox = top, agree[-1][2]
        conf = sum(v[1] for v in agree) / len(agree)

        out = YoloDetectionArray()
        out.header = msg.header
        d = YoloDetection()
        d.class_name = label
        d.confidence = float(conf)
        d.bbox = [float(v) for v in bbox]
        out.detections = [d]
        self.pub.publish(out)


def main():
    rclpy.init()
    n = BarCounterNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
