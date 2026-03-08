#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from actionlib_msgs.msg import GoalStatusArray, GoalID
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import math
import subprocess
import time
from collections import deque
import random
import json
import sys
import select
import termios
import tty
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Import the VLM detector
from pixel_to_cords import VLM_Response_Processor


# ============================================
# TELEOP SETTINGS
# ============================================
TELEOP_BINDINGS = {
    'w': (1, 0),   # Forward
    's': (-1, 0),  # Backward
    'a': (0, 1),   # Turn left
    'd': (0, -1),  # Turn right
    'q': (1, 1),   # Forward + left
    'e': (1, -1),  # Forward + right
    'z': (-1, 1),  # Backward + left
    'c': (-1, -1), # Backward + right
    ' ': (0, 0),   # Stop (spacebar)
}

TELEOP_HELP = """
============================================
🎮 TELEOP CONTROLS (with continuous detection)
============================================
   Q    W    E
   A    S    D
   Z         C

W/S : Forward/Backward
A/D : Turn Left/Right
Q/E : Forward + Turn
Z/C : Backward + Turn
SPACE : Stop

+/- : Increase/Decrease speed
M   : Show map stats
O   : Show detected objects
D   : Toggle VLM detection ON/OFF
H   : Show this help
X   : Exit program
============================================
VLM detection runs continuously while you drive!
============================================
"""


class KeyboardController:
    """Non-blocking keyboard input handler"""
    
    def __init__(self):
        self.settings = None
        if sys.stdin.isatty():
            self.settings = termios.tcgetattr(sys.stdin)
    
    def get_key(self, timeout=0.1):
        """Get keyboard input with timeout"""
        if not sys.stdin.isatty():
            return None
            
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
                return key
            return None
        except Exception:
            return None
        finally:
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
    
    def restore_terminal(self):
        """Restore terminal settings"""
        if self.settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class RobotTeleopExplorer:
    """
    Teleop-based exploration with continuous VLM detection.
    You drive the robot manually while detection runs automatically.
    """
    
    def __init__(self):
        
        # Launch system
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)
        
        script_dir = os.path.dirname(os.path.realpath(__file__)) 
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")
        
        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")
        
        subprocess.Popen(["roslaunch", "-p", port, launch_file])
        print("Gazebo system launched!")
        time.sleep(4.0)
        
        rospy.init_node('robot_teleop_explorer', anonymous=False)
        
        # Parameters
        self.camera_topic = "/realsense_camera/color/image_raw"
        self.pointcloud_topic = "/realsense_camera/depth/color/points"
        self.map_topic = "/map"
        self.odom_topic = "/r1/odom"
        self.save_dir = "/tmp/exploration_data"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Initialize VLM detector
        self.vlm_detector = VLM_Response_Processor(save_dir=self.save_dir)
        
        # Timing controls
        self.last_object_detection_time = rospy.Time.now().to_sec()
        self.object_detection_interval = 5.0  # seconds
        self.detection_enabled = True
       
        # File paths
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        
        if not os.path.exists(self.metadata_file):
            open(self.metadata_file, "w").close()
        
        # Sensor data
        self.current_pointcloud = None
        self.current_image = None
        self.current_pose = None
        
        # State variables
        self.frame_id = 0
        self.current_map = None
        self.map_data = None
        self.map_info = None
        
        # Detected objects tracking
        self.detected_objects_count = 0
        self.detected_objects_list = []

        # ============================================
        # TELEOP STATE
        # ============================================
        self.linear_speed = 0.5     # m/s
        self.angular_speed = 1.0    # rad/s
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.keyboard = KeyboardController()
        self.teleop_thread = None
        self.teleop_running = False
        self.program_running = True

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        
        # Synchronized subscribers for VLM detection
        image_sub = Subscriber(self.camera_topic, Image)
        pc_sub = Subscriber(self.pointcloud_topic, PointCloud2)
        odom_sub = Subscriber(self.odom_topic, Odometry)
        sync = ApproximateTimeSynchronizer(
            [image_sub, pc_sub, odom_sub], 
            queue_size=10, 
            slop=0.1
        )
        sync.registerCallback(self.synchronized_callback)
        
        # Print startup info
        self.print_startup_info()
        
        # Start teleop thread
        self.start_teleop_thread()
        
        # Velocity publishing timer (smooth control)
        rospy.Timer(rospy.Duration(0.1), self.publish_velocity)
        
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.spin()

    def print_startup_info(self):
        """Print startup information"""
        print("\n" + "="*60)
        print("🤖 TELEOP EXPLORATION WITH VLM DETECTION")
        print("="*60)
        print("Use WASD to drive the robot")
        print("VLM detection runs automatically every 5 seconds")
        print("Press 'H' for full controls help")
        print("="*60 + "\n")

    # ============================================
    # TELEOP METHODS
    # ============================================
    
    def start_teleop_thread(self):
        """Start the keyboard monitoring thread"""
        self.teleop_running = True
        self.teleop_thread = threading.Thread(target=self.teleop_loop, daemon=True)
        self.teleop_thread.start()
        rospy.loginfo("🎮 Teleop ready - use WASD to drive!")
    
    def stop_teleop_thread(self):
        """Stop the keyboard monitoring thread"""
        self.teleop_running = False
        if self.teleop_thread:
            self.teleop_thread.join(timeout=1.0)
        self.keyboard.restore_terminal()
    
    def teleop_loop(self):
        """Main teleop loop running in separate thread"""
        last_key_time = time.time()
        
        while self.teleop_running and not rospy.is_shutdown():
            try:
                key = self.keyboard.get_key(timeout=0.1)
                
                if key:
                    self.handle_key(key)
                    last_key_time = time.time()
                
                # Stop if no key pressed for a while
                if time.time() - last_key_time > 0.3:
                    self.current_linear = 0.0
                    self.current_angular = 0.0
                
            except Exception as e:
                rospy.logerr(f"Teleop error: {e}")
                time.sleep(0.1)
    
    def handle_key(self, key):
        """Handle keyboard input"""
        key_lower = key.lower()
        
        # Movement commands
        if key_lower in TELEOP_BINDINGS:
            linear, angular = TELEOP_BINDINGS[key_lower]
            self.current_linear = linear * self.linear_speed
            self.current_angular = angular * self.angular_speed
        
        # Speed control
        elif key == '+' or key == '=':
            self.linear_speed = min(self.linear_speed + 0.1, 1.0)
            self.angular_speed = min(self.angular_speed + 0.2, 2.0)
            rospy.loginfo(f"⬆️ Speed: linear={self.linear_speed:.1f}, angular={self.angular_speed:.1f}")
        elif key == '-':
            self.linear_speed = max(self.linear_speed - 0.1, 0.1)
            self.angular_speed = max(self.angular_speed - 0.2, 0.2)
            rospy.loginfo(f"⬇️ Speed: linear={self.linear_speed:.1f}, angular={self.angular_speed:.1f}")
        
        # Info commands
        elif key_lower == 'h':
            print(TELEOP_HELP)
        elif key_lower == 'm':
            self.print_map_stats()
        elif key_lower == 'o':
            self.print_detected_objects()
        elif key_lower == 'd':
            self.toggle_detection()
        elif key_lower == 'x' or key == '\x1b':  # X or ESC
            rospy.loginfo("👋 Exiting...")
            self.program_running = False
            rospy.signal_shutdown("User requested exit")
    
    def publish_velocity(self, event):
        """Publish velocity command (called by timer)"""
        cmd = Twist()
        cmd.linear.x = self.current_linear
        cmd.angular.z = self.current_angular
        self.cmd_vel_pub.publish(cmd)
    
    def toggle_detection(self):
        """Toggle VLM detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        rospy.loginfo(f"🔍 VLM Detection: {status}")
    
    def print_map_stats(self):
        """Print current map statistics"""
        if self.map_data is None:
            rospy.loginfo("📊 No map data available")
            return
        
        unknown = np.sum(self.map_data == -1)
        free = np.sum(self.map_data == 0)
        occupied = np.sum(self.map_data > 0)
        total = self.map_data.size
        explored = ((free + occupied) / total) * 100
        
        print("\n" + "="*50)
        print("📊 MAP STATISTICS")
        print("="*50)
        print(f"   Explored:       {explored:.1f}%")
        print(f"   Free cells:     {free}")
        print(f"   Occupied:       {occupied}")
        print(f"   Unknown:        {unknown}")
        if self.current_pose:
            print(f"   Robot position: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})")
        print(f"   Objects found:  {self.detected_objects_count}")
        print(f"   Detection:      {'ON' if self.detection_enabled else 'OFF'}")
        print("="*50 + "\n")
    
    def print_detected_objects(self):
        """Print list of detected objects"""
        print("\n" + "="*50)
        print("🎯 DETECTED OBJECTS")
        print("="*50)
        
        if not self.detected_objects_list:
            print("   No objects detected yet")
        else:
            for i, obj in enumerate(self.detected_objects_list[-10:], 1):  # Last 10
                print(f"   {i}. {obj['class']} at ({obj['x']:.2f}, {obj['y']:.2f})")
        
        print(f"\n   Total: {self.detected_objects_count} objects")
        print("="*50 + "\n")
    
    def shutdown_hook(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down...")
        self.stop_teleop_thread()
        # Stop robot
        cmd = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

    # ============================================
    # SENSOR CALLBACKS
    # ============================================

    def synchronized_callback(self, image_msg, pc_msg, odom_msg):
        """Handle synchronized sensor data - VLM detection while driving"""
        if not self.detection_enabled:
            return
            
        try:
            current_time = rospy.Time.now().to_sec()

            # Run VLM detection every N seconds (doesn't stop robot)
            if current_time - self.last_object_detection_time >= self.object_detection_interval:
                
                rospy.loginfo("🔍 Running VLM detection (robot keeps moving)...")
                
                # Capture image
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                
                # Run detection (non-blocking - robot continues moving)
                num_detected = self.vlm_detector.detect_objects(
                    cv_image, 
                    pc_msg,
                    odom_msg.pose.pose,
                    current_time,
                    self.frame_id
                )
                
                # Update detection count
                self.detected_objects_count += num_detected
                
                # Save metadata
                self.save_image_with_metadata(cv_image, current_time, odom_msg.pose.pose)
                
                if num_detected > 0:
                    rospy.loginfo(f"✅ Detected {num_detected} new objects! (Total: {self.detected_objects_count})")
                else:
                    rospy.loginfo(f"📷 Frame captured, no new objects")
                
                self.last_object_detection_time = current_time

        except Exception as e:
            rospy.logerr(f"Detection callback error: {e}")
    
    def save_image_with_metadata(self, cv_image, current_time, current_pose):
        """Save image with metadata"""
        metadata = {
            "frame_id": self.frame_id,
            "timestamp": current_time,
        }
        
        if current_pose and self.map_info:
            robot_x = current_pose.position.x
            robot_y = current_pose.position.y
            
            q = current_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            robot_theta = math.atan2(siny_cosp, cosy_cosp)
            
            metadata.update({
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_theta": robot_theta,
            })
        
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")
        
        self.frame_id += 1
    
    def map_callback(self, msg):
        """Handle map updates"""
        self.current_map = msg
        self.map_info = msg.info
        
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        if self.map_data is None or self.map_data.size == 0:
            return 
        
        # Save map image
        grid_img = np.zeros_like(self.map_data, dtype=np.uint8)
        grid_img[self.map_data == -1] = 127
        grid_img[self.map_data == 0] = 255
        grid_img[self.map_data > 0] = 0
        
        map_filename = os.path.join(self.save_dir, "map_latest.png")
        cv2.imwrite(map_filename, grid_img)
    
    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose


if __name__ == "__main__":
    try:
        RobotTeleopExplorer()
    except rospy.ROSInterruptException:
        pass
'''

## Key Features

| Feature | Description |
|---------|-------------|
| **Continuous Teleop** | WASD controls always active |
| **Background Detection** | VLM runs every 5 seconds while you drive |
| **Non-blocking** | Robot keeps moving during detection |
| **Speed Control** | +/- to adjust speed |
| **Info Display** | M for map stats, O for detected objects |
| **Toggle Detection** | D to turn detection on/off |

## Controls Summary
```
============================================
🎮 CONTROLS
============================================
MOVEMENT:
   Q    W    E        W = Forward
   A    S    D        S = Backward
   Z         C        A/D = Turn

SPEED:
   +/- = Increase/Decrease speed

INFO:
   M = Map statistics
   O = Detected objects list
   D = Toggle detection ON/OFF
   H = Help

EXIT:
   X or ESC = Exit program
============================================'''