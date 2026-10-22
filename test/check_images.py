#!/usr/bin/env python3
"""
Run the bar counter on still images, no ROS needed.

    python3 test/check_images.py test/images/*.jpg
"""
import os
import sys
import types

import cv2

# Stub the ROS imports so classify() can run on a laptop.
for m in ("rclpy", "rclpy.node", "dexi_interfaces", "dexi_interfaces.msg",
          "sensor_msgs", "sensor_msgs.msg"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["rclpy.node"].Node = object
sys.modules["dexi_interfaces.msg"].YoloDetection = None
sys.modules["dexi_interfaces.msg"].YoloDetectionArray = None
sys.modules["sensor_msgs.msg"].CompressedImage = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bar_counter_node import BarCounterNode  # noqa: E402

DEFAULTS = dict(min_disc_area=800, core_gap=50, min_ink=0.02, min_bar_frac=0.004,
                bar_aspect=1.2, min_circularity=0.70,
                max_circularity=1.05, max_bar_width_frac=0.22,
                bar_width_ratio=0.23, ink_tolerance=0.55)

node = BarCounterNode.__new__(BarCounterNode)
node.__dict__.update(DEFAULTS)

for path in sys.argv[1:]:
    label, conf, _bbox, note = node.classify(cv2.imread(path))
    print(f"{os.path.basename(path)}: {label} ({conf:.2f}) {note}")
