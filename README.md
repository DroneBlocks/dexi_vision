# dexi_vision

Basic OpenCV vision nodes for DEXI. Each node subscribes to the camera on
`/cam0/image_raw/compressed` and publishes what it finds, so it can be used from
Node-RED via rosbridge, from Python, or from any ROS 2 subscriber.

| Node | Publishes | What it finds | Docs |
|---|---|---|---|
| `color_detection_node.py` | `/color_detections` | Colored regions (HSV ranges) | [docs/color-detection.md](docs/color-detection.md) |
| `bar_counter_node.py` | `/bar_detections` | White bars in a black disc: `bars_0` to `bars_3` | [docs/bar-counter.md](docs/bar-counter.md) |

## Quick start

```bash
ros2 launch dexi_vision color_detection_launch.py
ros2 topic echo /color_detections

ros2 launch dexi_vision bar_counter_launch.py
ros2 topic echo /bar_detections
```

## Bar counter

`/bar_detections` uses `dexi_interfaces/msg/YoloDetectionArray`, the same message
as `/yolo_detections`. `class_name` is one of:

| class_name | Sees |
|---|---|
| `bars_1` | 1 white bar in a black disc |
| `bars_2` | 2 bars |
| `bars_3` | 3 bars |
| `bars_0` | plain black disc |

Nothing is published when the disc cannot be read (too far, tilted, bars
merged), so no message means "not sure", not `bars_0`. An answer is only
published once it wins 3 of the last 5 frames. The node runs at 2 Hz by
default and reads reliably from 0.5 to 1.0 m above a 3 in disc.

All thresholds are ROS parameters in `config/bar_counter_params.yaml`.
[docs/bar-counter.md](docs/bar-counter.md) walks through each step with pictures.

To test on still images without ROS:

```bash
python3 test/check_images.py test/images/*.jpg
```

## Dependencies

- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `dexi_interfaces` (for `ColorDetection` / `ColorDetectionArray` messages)
- `python3-opencv` (`cv2`)
- `numpy`

The bar counter was written by Aziz Beghdadi for the AVR 2026 bridge cups.
