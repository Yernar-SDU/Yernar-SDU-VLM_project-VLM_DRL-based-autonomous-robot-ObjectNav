#!/usr/bin/env python3
"""
Simple Camera Calibration Test
Usage: Place marker at known distance, run test, adjust calibration parameters
"""

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import math
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from actionlib_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import cv2
import os
import numpy as np
import math
import subprocess
import time
from collections import deque
import random
import json
import base64
import tf.transformations as tf_trans
from openai import OpenAI
import os
# VLM import
class SimpleCalibrationTest:
    def __init__(self):
        # Launch system
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)
        rospy.init_node('simple_calibration_test')
        
        self.pointcloud = None
        self.pose = None


        script_dir = os.path.dirname(os.path.realpath(__file__))
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")
        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")
                
        subprocess.Popen(["roslaunch", "-p", port, launch_file])
        print("LiDAR system launched!")
        time.sleep(5.0)


        
        rospy.Subscriber("/realsense_camera/depth/color/points", PointCloud2, 
                        lambda msg: setattr(self, 'pointcloud', msg))
        rospy.Subscriber("/r1/odom", Odometry, 
                        lambda msg: setattr(self, 'pose', msg.pose.pose))
        
        # Current calibration parameters
        self.cam_offset = np.array([0.0, 0.0, 0.3])  # [forward, left, up]
        self.cam_tilt = 0.0  # radians
        
        print("\n" + "="*70)
        print("CAMERA CALIBRATION TESTER")
        print("="*70)
        print("\nWaiting for data...")
        
        rospy.sleep(2.0)
        
        if self.pointcloud and self.pose:
            print("Ready!\n")
            self.print_instructions()
        else:
            print("ERROR: No data received. Check topics are publishing.")

        time.sleep(10)
        self.update_calibration(forward=-0.74, left=0.0, up=0.3)
        self.quick_test(320, 240, distance=1.0)
    
    def print_instructions(self):
        print("\nQUICK START:")
        print("1. Place a marker 1.0m directly in front of robot")
        print("2. Open: rosrun rqt_image_view rqt_image_view")
        print("3. Select topic: /realsense_camera/color/image_raw")
        print("4. Hover over marker, note pixel coordinates (shown at bottom)")
        print("5. In Python console, run:")
        print("   >>> tester.quick_test(pixel_x, pixel_y, distance=1.0)")
        print("\nExample:")
        print("   >>> tester.quick_test(320, 240, distance=1.0)")
        print("="*70 + "\n")
    
    def quick_test(self, pixel_x, pixel_y, distance=1.0, lateral_offset=0.0):
        """
        Quick test for marker at known distance ahead of robot
        
        Args:
            pixel_x, pixel_y: Pixel coordinates from image
            distance: How far ahead of robot (meters)
            lateral_offset: Left/right offset (meters, positive = left)
        """
        if not self.pointcloud or not self.pose:
            print("ERROR: No sensor data")
            return
        
        # Calculate expected position
        yaw = self.get_yaw()
        expected = np.array([
            self.pose.position.x + distance * math.cos(yaw) - lateral_offset * math.sin(yaw),
            self.pose.position.y + distance * math.sin(yaw) + lateral_offset * math.cos(yaw),
            0.0
        ])
        
        # Get detection
        detected = self.detect_position(pixel_x, pixel_y)
        
        if detected is None:
            print("ERROR: No valid point cloud data at that pixel")
            return
        
        # Results
        error = np.linalg.norm(detected[:2] - expected[:2])
        
        print("\n" + "-"*70)
        print(f"CALIBRATION TEST RESULTS")
        print("-"*70)
        print(f"Pixel: ({pixel_x}, {pixel_y})")
        print(f"Expected position: ({expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f})")
        print(f"Detected position: ({detected[0]:.3f}, {detected[1]:.3f}, {detected[2]:.3f})")
        print(f"\nError: {error:.3f}m")
        
        if error < 0.05:
            print("Status: EXCELLENT")
        elif error < 0.15:
            print("Status: GOOD")
        elif error < 0.30:
            print("Status: NEEDS TUNING")
        else:
            print("Status: POOR - RECALIBRATE")
            print("\nSuggestions:")

            
            dx = detected[0] - expected[0]
            dy = detected[1] - expected[1]
            
            if abs(dx) > 0.1:
                print(f"  - Forward/back error: {dx:+.3f}m")
                print(f"    Try: cam_offset[0] = {self.cam_offset[0] - dx:.3f}")
            if abs(dy) > 0.1:
                print(f"  - Left/right error: {dy:+.3f}m")
                print(f"    Try: cam_offset[1] = {self.cam_offset[1] - dy:.3f}")
        
        print("-"*70 + "\n")
        
        return {'expected': expected, 'detected': detected, 'error': error}
    
    def detect_position(self, px, py):
        """Get world position from pixel coordinates"""
        # Sample neighborhood for robustness
        samples = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pts = list(pc2.read_points(
                    self.pointcloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                    uvs=[(px + dx, py + dy)]
                ))
                if pts:
                    x, y, z = pts[0]
                    if not math.isnan(x) and 0.05 < z < 10.0:
                        samples.append([x, y, z])
        
        if not samples:
            return None
        
        # Median filter
        cam_coords = np.median(samples, axis=0)
        
        # Transform to world
        return self.cam_to_world(*cam_coords)
    
    def cam_to_world(self, cx, cy, cz):
        """Transform camera coords to world coords"""
        # Camera to robot frame
        robot_frame = np.array([cz, -cx, -cy]) + self.cam_offset
        
        # Robot to world frame
        yaw = self.get_yaw()
        R = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        world_offset = R @ robot_frame
        
        return np.array([
            self.pose.position.x + world_offset[0],
            self.pose.position.y + world_offset[1],
            self.pose.position.z + world_offset[2]
        ])
    
    def get_yaw(self):
        """Get robot yaw angle"""
        q = self.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    
    def update_calibration(self, forward=None, left=None, up=None):
        """Update calibration parameters"""
        if forward is not None:
            self.cam_offset[0] = forward
        if left is not None:
            self.cam_offset[1] = left
        if up is not None:
            self.cam_offset[2] = up
        
        print(f"Updated calibration: {self.cam_offset}")

if __name__ == "__main__":
    tester = SimpleCalibrationTest()

    # Keep node alive for interactive testing
    print("Node running. Use Python console to call tester.quick_test()")
    print("Press Ctrl+C to exit\n")
    rospy.spin()