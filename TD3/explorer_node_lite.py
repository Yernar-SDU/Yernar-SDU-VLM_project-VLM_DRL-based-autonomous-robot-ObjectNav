#!/usr/bin/env python3
"""
Explorer Node (Lite)
Handles VLM object detection and data saving only.
Robot exploration is delegated to explore_lite running in a separate terminal:
  roslaunch explore_lite explore.launch
"""
import rospy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from cv_bridge import CvBridge
import cv2
import os
import json
import math
import subprocess
import time
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber

from pixel_to_cords import VLM_Response_Processor


class RobotExplorerLite:

    def __init__(self, vlm_type='moondream', world='TD3_signs2.world'):
        self.vlm_type = vlm_type

        # Launch roscore + Gazebo
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")

        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")

        print(f"Launching world: {world}")
        subprocess.Popen(["roslaunch", "-p", port, launch_file, f"world_name:={world}"])
        print("Gazebo launched!")
        time.sleep(4.0)

        rospy.init_node('robot_explorer_lite', anonymous=False)

        # Topics
        self.camera_topic = "/realsense_camera/color/image_raw"
        self.pointcloud_topic = "/realsense_camera/depth/color/points"
        self.map_topic = "/map"
        self.odom_topic = "/r1/odom"
        self.save_dir = "/tmp/exploration_data"

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.vlm_detector = VLM_Response_Processor(self.vlm_type, save_dir=self.save_dir)

        # Detection timing
        self.last_object_detection_time = rospy.Time.now().to_sec()
        self.object_detection_interval = 2  # seconds

        # Metadata file
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        if not os.path.exists(self.metadata_file):
            open(self.metadata_file, "w").close()

        # Sensor state
        self.current_pose = None
        self.map_info = None
        self.frame_id = 0

        # Subscribers
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        # Synchronized image + pointcloud + odom for detection
        image_sub = Subscriber(self.camera_topic, Image)
        pc_sub = Subscriber(self.pointcloud_topic, PointCloud2)
        odom_sub = Subscriber(self.odom_topic, Odometry)
        sync = ApproximateTimeSynchronizer(
            [image_sub, pc_sub, odom_sub],
            queue_size=100,
            slop=0.01
        )
        sync.registerCallback(self.synchronized_callback)

        rospy.loginfo("=" * 60)
        rospy.loginfo("ROBOT EXPLORER LITE — VLM detection only")
        rospy.loginfo("Exploration handled by explore_lite (separate terminal)")
        rospy.loginfo("=" * 60)

        rospy.spin()

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        self.map_info = msg.info

        # Save latest map image for reference
        import numpy as np
        map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        grid_img = np.zeros_like(map_data, dtype=np.uint8)
        grid_img[map_data == -1] = 127   # unknown = gray
        grid_img[map_data == 0] = 255    # free   = white
        grid_img[map_data > 0] = 0       # occupied = black
        cv2.imwrite(os.path.join(self.save_dir, "map_latest.png"), grid_img)

    def synchronized_callback(self, image_msg, pc_msg, odom_msg):
        """Periodic VLM detection on synchronized sensor data."""
        try:
            current_time = rospy.Time.now().to_sec()

            if int(current_time) % int(self.object_detection_interval) != 0:
                return
            if current_time - self.last_object_detection_time < self.object_detection_interval:
                return

            self.last_object_detection_time = current_time
            start_ts = current_time

            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            pose = odom_msg.pose.pose

            def run_detection(img, pc, p, ts):
                num_detected = self.vlm_detector.detect_objects(img, pc, p, ts, self.frame_id)
                diff = rospy.Time.now().to_sec() - ts
                with open("latencies.txt", "a") as f:
                    f.write(f"Time: {ts}, Latency: {diff:.4f}s, Count: {num_detected}\n")
                rospy.loginfo(f"VLM detected {num_detected} object(s) in {diff:.2f}s")
                self.save_image_with_metadata(img, ts, p)

            threading.Thread(
                target=run_detection,
                args=(cv_image, pc_msg, pose, start_ts),
                daemon=True
            ).start()

        except Exception as e:
            rospy.logerr(f"Detection error: {e}")

    # ------------------------------------------------------------------
    # Data saving
    # ------------------------------------------------------------------

    def save_image_with_metadata(self, cv_image, current_time, current_pose):
        metadata = {"timestamp": current_time}

        if current_pose and self.map_info:
            robot_x = current_pose.position.x
            robot_y = current_pose.position.y

            q = current_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            robot_theta = math.atan2(siny_cosp, cosy_cosp)

            map_cell_x = int((robot_x - self.map_info.origin.position.x) / self.map_info.resolution)
            map_cell_y = int((robot_y - self.map_info.origin.position.y) / self.map_info.resolution)

            metadata.update({
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_theta": robot_theta,
                "map_origin_x": self.map_info.origin.position.x,
                "map_origin_y": self.map_info.origin.position.y,
                "map_resolution": self.map_info.resolution,
                "map_cell_x": map_cell_x,
                "map_cell_y": map_cell_y,
            })
        else:
            rospy.logwarn_throttle(10, "Saving frame without pose/map metadata")

        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        self.frame_id += 1
        rospy.loginfo(f"Saved frame {self.frame_id} at t={current_time:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explorer Node Lite — detection only")
    parser.add_argument("--vlm", "-a", type=str, default="moondream",
                        choices=["moondream", "gpt4o"],
                        help="VLM backend for object detection")
    parser.add_argument("--world", "-w", type=str, default="TD3_signs2.world",
                        help="Gazebo world file")
    args = parser.parse_args()

    try:
        RobotExplorerLite(args.vlm, args.world)
    except rospy.ROSInterruptException:
        pass
