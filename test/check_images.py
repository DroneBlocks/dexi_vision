#!/usr/bin/env python3
"""
Run the bar counter on still images, no ROS needed.

    python3 test/check_images.py test/images/*.jpg
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bar_counter import BarCounter  # noqa: E402

counter = BarCounter()
for path in sys.argv[1:]:
    r = counter.read(cv2.imread(path))
    print(f"{os.path.basename(path)}: {r.label} ({r.confidence:.2f}) {r.note}")
