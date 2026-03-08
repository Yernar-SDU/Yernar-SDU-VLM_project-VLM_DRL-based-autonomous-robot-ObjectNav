#!/usr/bin/env python3
"""
Coordinate Frame Verification Test
Tests if camera optical frame -> robot base frame -> world frame transformations are correct
"""

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import math

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

class CoordinateFrameVerifier:
    def __init__(self):
        # Launch system
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)
        self.pointcloud = None
        self.pose = None


        script_dir = os.path.dirname(os.path.realpath(__file__))
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")
        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")
                
        subprocess.Popen(["roslaunch", "-p", port, launch_file])
        print("LiDAR system launched!")
        time.sleep(5.0)


        rospy.init_node('coordinate_frame_verifier')
        
        self.pointcloud = None
        self.pose = None
        
        rospy.Subscriber("/realsense_camera/depth/color/points", PointCloud2,
                        lambda msg: setattr(self, 'pointcloud', msg))
        rospy.Subscriber("/r1/odom", Odometry,
                        lambda msg: setattr(self, 'pose', msg.pose.pose))
        
        # Calibrated values
        self.cam_offset = np.array([0.26, 0.0, 0.3])
        self.cam_tilt = 0.0
        
        print("\n" + "="*70)
        print("COORDINATE FRAME VERIFICATION")
        print("="*70)
        
        rospy.sleep(2.0)
        
        if not self.pointcloud or not self.pose:
            print("ERROR: No sensor data")
            return
        
        self.run_verification()



        rospy.sleep(5)
        self.test_consistency(320, 240, "test_marker")
        rospy.sleep(5)
        self.test_consistency(400, 250, "test_marker")
    
    def run_verification(self):
        """Run comprehensive frame verification tests"""
        
        print("\nTEST 1: Frame Convention Check")
        print("-" * 70)
        self.test_frame_convention()
        
        print("\n\nTEST 2: Multi-Position Consistency")
        print("-" * 70)
        print("Place a marker, detect it, then move robot and detect again.")
        print("Object world coordinates should remain constant.")
        print("Call: verifier.test_consistency(pixel_x, pixel_y)")
        
        print("\n\nTEST 3: Known Distance Grid")
        print("-" * 70)
        print("Place markers at grid positions (0.5m, 1.0m, 1.5m forward)")
        print("Verify all distances are detected accurately")
    
    def test_frame_convention(self):
        """Verify RealSense optical frame convention"""
        print("\nRealSense camera optical frame convention:")
        print("  X-axis: Points RIGHT in image")
        print("  Y-axis: Points DOWN in image")
        print("  Z-axis: Points FORWARD (depth)")
        
        print("\nRobot base frame convention (REP-103):")
        print("  X-axis: Points FORWARD")
        print("  Y-axis: Points LEFT")
        print("  Z-axis: Points UP")
        
        print("\nExpected transformation:")
        print("  robot_X = camera_Z (depth becomes forward)")
        print("  robot_Y = -camera_X (right becomes left)")
        print("  robot_Z = -camera_Y (down becomes up)")
        
        print("\nYour current transformation:")
        print(f"  cam_offset = {self.cam_offset}")
        print(f"  Mapping: [cam_z, -cam_x, -cam_y] + offset")
        
        # Test center pixel
        width = 640  # Typical RealSense width
        height = 480  # Typical RealSense height
        center_x, center_y = width // 2, height // 2
        
        print(f"\nSampling center pixel ({center_x}, {center_y})...")
        result = self.detect_position(center_x, center_y)
        
        if result:
            cam_coords, world_coords = result
            print(f"  Camera frame: ({cam_coords[0]:.3f}, {cam_coords[1]:.3f}, {cam_coords[2]:.3f})")
            print(f"  World frame: ({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
            
            # Verify object is in front of robot
            robot_to_obj_x = world_coords[0] - self.pose.position.x
            robot_to_obj_y = world_coords[1] - self.pose.position.y
            
            yaw = self.get_yaw()
            forward_component = robot_to_obj_x * math.cos(yaw) + robot_to_obj_y * math.sin(yaw)
            lateral_component = -robot_to_obj_x * math.sin(yaw) + robot_to_obj_y * math.cos(yaw)
            
            print(f"\nRelative to robot:")
            print(f"  Forward: {forward_component:.3f}m")
            print(f"  Lateral: {lateral_component:.3f}m")
            
            if forward_component > 0.1 and abs(lateral_component) < 0.3:
                print("  Status: CORRECT - Object in front of robot")
            elif forward_component < -0.1:
                print("  Status: ERROR - Object behind robot (frame reversed?)")
            elif abs(lateral_component) > 0.5:
                print("  Status: ERROR - Object to the side (X/Y swapped?)")
    
    def test_consistency(self, pixel_x, pixel_y, marker_name="object"):
        """
        Test if same object maintains consistent world coordinates from different robot positions
        
        Usage:
        1. Place marker, note pixel coordinates
        2. Run: verifier.test_consistency(px, py, "marker1")
        3. Move robot to different position
        4. Run again with same pixel coordinates
        5. World coordinates should be similar (within 10cm)
        """
        result = self.detect_position(pixel_x, pixel_y)
        if not result:
            print("ERROR: Could not detect position")
            return
        
        cam_coords, world_coords = result
        
        # Save detection
        if not hasattr(self, 'consistency_tests'):
            self.consistency_tests = {}
        
        if marker_name not in self.consistency_tests:
            self.consistency_tests[marker_name] = []
        
        detection = {
            'world': world_coords,
            'robot_pos': np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z]),
            'robot_yaw': self.get_yaw()
        }
        
        self.consistency_tests[marker_name].append(detection)
        
        print(f"\nDetection #{len(self.consistency_tests[marker_name])} for '{marker_name}':")
        print(f"  Robot at: ({detection['robot_pos'][0]:.3f}, {detection['robot_pos'][1]:.3f})")
        print(f"  Object at: ({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
        
        # Analyze consistency if multiple detections
        if len(self.consistency_tests[marker_name]) >= 2:
            detections = self.consistency_tests[marker_name]
            positions = np.array([d['world'] for d in detections])
            
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            max_deviation = np.max(np.linalg.norm(positions - mean_pos, axis=1))
            
            print(f"\nConsistency Analysis ({len(detections)} detections):")
            print(f"  Mean position: ({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")
            print(f"  Std deviation: ({std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f})")
            print(f"  Max deviation: {max_deviation:.3f}m")
            
            if max_deviation < 0.1:
                print("  Status: EXCELLENT - Coordinates very consistent")
            elif max_deviation < 0.2:
                print("  Status: GOOD - Acceptable consistency")
            elif max_deviation < 0.5:
                print("  Status: NEEDS TUNING - Significant variance")
            else:
                print("  Status: POOR - Coordinate system likely incorrect")
                print("\n  Possible issues:")
                print("    - Camera frame convention wrong")
                print("    - Robot odometry drifting")
                print("    - Calibration offset incorrect")
    
    def detect_position(self, px, py):
        """Get camera and world coordinates from pixel"""
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
        
        cam_coords = np.median(samples, axis=0)
        world_coords = self.cam_to_world(*cam_coords)
        
        return (cam_coords, world_coords)
    
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

if __name__ == "__main__":
    verifier = CoordinateFrameVerifier()
    
    print("\n\nInteractive commands:")
    print("  verifier.test_consistency(pixel_x, pixel_y, 'marker_name')")
    print("\nPress Ctrl+C to exit")
    rospy.spin()