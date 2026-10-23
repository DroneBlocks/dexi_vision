#!/usr/bin/env python3
"""
Bar counter node.

Counts the white bars inside the black disc under the camera and publishes
the count on /bar_detections as a YoloDetectionArray, so it reads like any
other detector:

    1 bar  -> bars_1
    2 bars -> bars_2
    3 bars -> bars_3
    0 bars -> bars_0   (plain black disc)

Nothing is published when the disc cannot be read (too far, tilted, bars
merged). No message means "not sure", not "no bars".

The node knows nothing about the game. On the AVR 2026 field the discs are the
bridge-cup stickers, and the Node-RED flow maps bars_1/2/3 to red/green/blue.
The computer vision is in bar_counter.py; this file adds the ROS topics, the
detection_frequency gate and the vote.
"""
from collections import Counter, deque
from dataclasses import asdict

import cv2
import numpy as np
import rclpy
from dexi_interfaces.msg import YoloDetection, YoloDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from bar_counter import BarCounter, Params


class BarCounterNode(Node):
    def __init__(self):
        super().__init__("bar_counter_node")

        tuning = asdict(Params())
        for name, default in tuning.items():
            self.declare_parameter(name, default)
        self.declare_parameter("detection_frequency", 2.0)
        self.declare_parameter("vote_window", 5)
        self.declare_parameter("vote_min", 3)
        self.declare_parameter("verbose", False)

        self.counter = BarCounter(Params(**{n: self.get_parameter(n).value for n in tuning}))
        self.freq = self.get_parameter("detection_frequency").value
        self.min_interval = 1.0 / max(0.1, self.freq)
        self.votes = deque(maxlen=self.get_parameter("vote_window").value)
        self.vote_min = self.get_parameter("vote_min").value
        self.verbose = self.get_parameter("verbose").value
        self.last_t = 0.0

        self.pub = self.create_publisher(YoloDetectionArray, "/bar_detections", 10)
        self.create_subscription(CompressedImage, "/cam0/image_raw/compressed", self.cb, 10)
        self.get_logger().info(f"bar_counter_node publishing /bar_detections at {self.freq} Hz")

    def cb(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_t < self.min_interval:
            return
        self.last_t = now

        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        reading = self.counter.read(bgr)
        if self.verbose:
            self.get_logger().info(f"{reading.label} ({reading.confidence:.2f}) {reading.note}")

        # Publish only when one answer wins vote_min of the last vote_window
        # frames. Abstentions vote too, so a flickering read publishes nothing.
        self.votes.append(reading)
        top, count = Counter(r.label for r in self.votes).most_common(1)[0]
        if top is None or count < self.vote_min:
            return
        agree = [r for r in self.votes if r.label == top]

        d = YoloDetection()
        d.class_name = top
        d.confidence = float(sum(r.confidence for r in agree) / len(agree))
        d.bbox = [float(v) for v in agree[-1].bbox]
        out = YoloDetectionArray()
        out.header = msg.header
        out.detections = [d]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = BarCounterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
