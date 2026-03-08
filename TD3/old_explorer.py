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
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

class VLMObjectDetectionExplorer:
    def __init__(self):
        
        # Launch system
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)
        self.client = openai.OpenAI(api_key="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")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")
        
        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")
        
        subprocess.Popen(["roslaunch", "-p", port, launch_file])
        print("LiDAR system launched!")
        time.sleep(5.0)
        
        rospy.init_node('vlm_object_detection_explorer', anonymous=False)
        
        # Parameters
        self.camera_topic = "/realsense_camera/color/image_raw"
        self.pointcloud_topic = "/realsense_camera/depth/color/points"
        self.camera_info_topic = "/realsense_camera/color/camera_info"
        self.map_topic = "/map"
        self.odom_topic = "/r1/odom"
        self.save_dir = "/tmp/exploration_data"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Timing controls
        self.last_image_save_time = rospy.Time.now().to_sec()
        self.last_object_detection_time = rospy.Time.now().to_sec()
        self.image_save_interval = 10.0  # seconds
        self.object_detection_interval = 5.0  # seconds
        
        # File paths
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        self.objects_file = os.path.join(self.save_dir, "detected_objects.jsonl")
        
        # Create files if they don't exist
        for file_path in [self.metadata_file, self.objects_file]:
            if not os.path.exists(file_path):
                open(file_path, "w").close()

        # VLM setup
        self.setup_vlm()
        
        # Sensor data
        self.current_pointcloud = None
        self.camera_info = None
        self.detected_objects = []
        
        # Publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.camera_callback)
        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        
        # State variables
        self.frame_id = 0
        self.object_id = 0
        self.current_map = None
        self.current_pose = None
        self.map_data = None
        self.map_info = None
        
        # Navigation state
        self.current_goal_status = None
        self.goal_reached = True
        self.stuck_counter = 0
        self.max_stuck_count = 2
        
        # Position tracking for stuck detection
        self.position_history = deque(maxlen=30)
        self.last_position_check = 0
        self.min_movement_threshold = 0.05
        
        # Goal management with proper blacklisting
        self.current_goal = None
        self.failed_goals = set()
        self.goal_attempt_count = {}
        self.max_attempts_per_goal = 2
        self.blacklist_radius = 1.5
        
        # Frontier selection strategy
        self.last_selected_frontier_index = 0
        self.frontier_selection_strategy = "round_robin"
        
        # Exploration state
        self.exploration_active = False
        self.last_goal_time = 0
        self.goal_timeout = 12.0
        self.recovery_mode = False
        self.consecutive_same_goals = 0
        self.last_goal_coords = None
        
        rospy.loginfo("="*60)
        rospy.loginfo("VLM OBJECT DETECTION EXPLORER")
        rospy.loginfo("="*60)
        rospy.loginfo("Features: VLM object detection, Point cloud coordinate estimation")
        
        # Main control loop
        rospy.Timer(rospy.Duration(0.5), self.exploration_control)
        rospy.Timer(rospy.Duration(1.5), self.position_check)
        rospy.Timer(rospy.Duration(3.0), self.cycle_detection)
        
        rospy.spin()
    
    def setup_vlm(self):
        """Setup VLM for object detection"""
        self.use_vlm = False
        
        if OPENAI_AVAILABLE:
            # Set your OpenAI API key here
            # openai.api_key = "your-api-key-here"
            self.use_vlm = True
            rospy.loginfo("📝 Set your OpenAI API key in setup_vlm() to enable VLM detection")
        
        if not self.use_vlm:
            rospy.logwarn("⚠️ VLM not configured. Set API key to enable object detection.")
    
    def camera_callback(self, msg):
        try:
            current_time = rospy.Time.now().to_sec()
            
            # Convert ROS Image to CV2
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Object detection with VLM (less frequent than image saving)
            if current_time - self.last_object_detection_time >= self.object_detection_interval:
                if self.use_vlm:
                    self.detect_and_save_objects(cv_image, current_time)
                self.last_object_detection_time = current_time
            
            # Regular image saving
            if current_time - self.last_image_save_time >= self.image_save_interval:
                self.save_image_with_metadata(cv_image, current_time)
                self.last_image_save_time = current_time

        except Exception as e:
            rospy.logerr(f"Camera callback error: {e}")
    
    def pointcloud_callback(self, msg):
        """Handle point cloud data"""
        self.current_pointcloud = msg
    
    def camera_info_callback(self, msg):
        """Handle camera info for intrinsic parameters"""
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo("📷 Camera info received")
    
    def save_image_with_metadata(self, cv_image, current_time):
        """Save image with existing metadata format"""
        filename = os.path.join(self.save_dir, f"frame_{self.frame_id:05d}.png")
        cv2.imwrite(filename, cv_image)

        # Save metadata
        if self.current_pose and self.map_info:
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y

            # Get yaw (theta) from quaternion
            q = self.current_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            robot_theta = math.atan2(siny_cosp, cosy_cosp)

            # Map cell coordinates
            map_cell_x = int((robot_x - self.map_info.origin.position.x) / self.map_info.resolution)
            map_cell_y = int((robot_y - self.map_info.origin.position.y) / self.map_info.resolution)

            metadata = {
                "id": f"img_{self.frame_id:06d}",
                "timestamp": current_time,
                "image_path": filename,
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_theta": robot_theta,
                "map_origin_x": self.map_info.origin.position.x,
                "map_origin_y": self.map_info.origin.position.y,
                "map_resolution": self.map_info.resolution,
                "map_cell_x": map_cell_x,
                "map_cell_y": map_cell_y
            }

            # Append to JSONL
            with open(self.metadata_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")

        self.frame_id += 1
        rospy.loginfo(f"Saved frame {self.frame_id} + metadata at {current_time:.1f}s")
    
    def detect_and_save_objects(self, cv_image, current_time):
        """Detect objects using VLM and save with 3D coordinates"""
        if self.current_pose is None or not self.use_vlm:
            return
        
        # Detect objects with VLM
        detected_objects = self.detect_with_vlm(cv_image)
        
        # Process each detected object
        for obj in detected_objects:
            # Get 3D coordinates from point cloud
            world_coords = self.get_coords_from_pointcloud(obj['center_x'], obj['center_y'])
            
            if world_coords:
                self.save_object_detection(obj, world_coords, current_time)
    
    def detect_with_vlm(self, image):
        """Object detection using Vision Language Model"""
        if not self.use_vlm:
            return []
        
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()

            response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify all objects in this image. For each object, provide: object_name:[x,y] where x,y are pixel coordinates of the center. Format each as 'object_name: [x, y]' on separate lines. Be specific about object types (e.g., 'chair', 'table', 'person', 'bottle', etc.)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=300
            )
            
            return self.parse_vlm_response(response.choices[0].message.content)
            
        except Exception as e:
            rospy.logerr(f"VLM detection error: {e}")
            return []
    
    def parse_vlm_response(self, response_text):
        """Parse VLM response to extract objects and coordinates"""
        objects = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            try:
                if ':' in line and '[' in line and ']' in line:
                    parts = line.split(':')
                    object_name = parts[0].strip()
                    coords_str = parts[1].strip()
                    
                    # Extract coordinates
                    coords_str = coords_str.replace('[', '').replace(']', '')
                    coords = coords_str.split(',')
                    if len(coords) == 2:
                        x, y = map(int, coords)
                        
                        obj = {
                            'class': object_name,
                            'confidence': 0.8,  # VLM doesn't provide confidence
                            'center_x': x,
                            'center_y': y,
                            'bbox': [x-25, y-25, x+25, y+25],  # Estimated bbox
                            'detection_method': 'vlm'
                        }
                        objects.append(obj)
            except Exception as e:
                rospy.logwarn(f"Failed to parse line: {line}, error: {e}")
                continue
        
        return objects
    
    def get_coords_from_pointcloud(self, pixel_x, pixel_y):
        """Extract 3D coordinates from point cloud at pixel location"""
        if not self.current_pointcloud:
            return None
            
        try:
            # Get point from point cloud at pixel location
            points = list(pc2.read_points(
                self.current_pointcloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
                uvs=[(pixel_x, pixel_y)]
            ))
            
            if points and len(points) > 0:
                x, y, z = points[0]
                
                # Check if valid (not NaN or too far)
                if not (math.isnan(x) or math.isnan(y) or math.isnan(z)) and z < 10.0:
                    # Transform to world coordinates
                    return self.transform_to_world_coords(x, y, z)
            
        except Exception as e:
            rospy.logwarn(f"Point cloud coordinate extraction error: {e}")
        
        return None
    
    def transform_to_world_coords(self, camera_x, camera_y, camera_z):
        """Transform camera coordinates to world coordinates"""
        if self.current_pose is None:
            return None
        
        try:
            # Get robot position and orientation
            robot_pos = self.current_pose.position
            robot_quat = self.current_pose.orientation
            
            # Convert quaternion to rotation matrix
            quat = [robot_quat.x, robot_quat.y, robot_quat.z, robot_quat.w]
            rotation_matrix = tf_trans.quaternion_matrix(quat)
            
            # Camera coordinates (assuming camera is forward-facing)
            # Adjust coordinate frame: ROS camera frame (x=forward, y=left, z=up)
            camera_point = np.array([camera_z, -camera_x, -camera_y, 1])
            
            # Transform to world coordinates
            world_point = np.dot(rotation_matrix, camera_point)
            
            # Add robot position
            world_x = robot_pos.x + world_point[0]
            world_y = robot_pos.y + world_point[1]
            world_z = robot_pos.z + world_point[2]
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            rospy.logwarn(f"Coordinate transformation error: {e}")
            return None
    
    def save_object_detection(self, obj, world_coords, timestamp):
        """Save detected object with metadata"""
        try:
            object_metadata = {
                "id": f"obj_{self.object_id:06d}",
                "timestamp": timestamp,
                "object_class": obj['class'],
                "confidence": obj['confidence'],
                "detection_method": obj['detection_method'],
                "world_x": world_coords[0],
                "world_y": world_coords[1],
                "world_z": world_coords[2],
                "pixel_x": obj['center_x'],
                "pixel_y": obj['center_y'],
                "bbox": obj['bbox'],
                "robot_x": self.current_pose.position.x,
                "robot_y": self.current_pose.position.y,
                "robot_z": self.current_pose.position.z,
                "robot_theta": self.get_robot_yaw(),
                "frame_id": self.frame_id
            }
            
            # Append to objects JSONL file
            with open(self.objects_file, "a") as f:
                f.write(json.dumps(object_metadata) + "\n")
            
            self.object_id += 1
            self.detected_objects.append(object_metadata)
            
            rospy.loginfo(f"🎯 Detected {obj['class']} at world coords: ({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f})")
            
        except Exception as e:
            rospy.logerr(f"Object saving error: {e}")
    
    def get_robot_yaw(self):
        """Get robot yaw angle from quaternion"""
        if self.current_pose is None:
            return 0.0
        
        q = self.current_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    # Object retrieval methods
    def find_objects_near_location(self, x, y, radius=2.0, object_class=None):
        """Find objects near a specific location"""
        objects = []
        try:
            with open(self.objects_file, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        distance = math.sqrt((obj["world_x"] - x)**2 + (obj["world_y"] - y)**2)
                        if distance <= radius:
                            if object_class is None or obj["object_class"] == object_class:
                                objects.append(obj)
        except FileNotFoundError:
            pass
        
        return objects
    
    def get_objects_by_class(self, object_class):
        """Retrieve all objects of a specific class"""
        objects = []
        try:
            with open(self.objects_file, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj["object_class"] == object_class:
                            objects.append(obj)
        except FileNotFoundError:
            pass
        
        return objects
    
    def get_recent_objects(self, time_window=300):  # 5 minutes
        """Get objects detected within time window"""
        current_time = rospy.Time.now().to_sec()
        objects = []
        
        try:
            with open(self.objects_file, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if current_time - obj["timestamp"] <= time_window:
                            objects.append(obj)
        except FileNotFoundError:
            pass
        
        return objects
    
    # Navigation and exploration methods (keeping original functionality)
    def map_callback(self, msg):
        self.current_map = msg
        self.map_info = msg.info
        
        # Convert map data to numpy array
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        
        # Save map image
        grid_img = np.zeros_like(self.map_data, dtype=np.uint8)
        grid_img[self.map_data == -1] = 127  # Unknown = gray
        grid_img[self.map_data == 0] = 255   # Free = white
        grid_img[self.map_data > 0] = 0      # Occupied = black
        
        map_filename = os.path.join(self.save_dir, "map_latest.png")
        cv2.imwrite(map_filename, grid_img)
        
        self.analyze_map()
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        # Track position history for stuck detection
        if self.current_pose:
            current_pos = (self.current_pose.position.x, self.current_pose.position.y)
            self.position_history.append((current_pos, rospy.Time.now().to_sec()))
    
    def goal_status_callback(self, msg):
        """Monitor navigation goal status"""
        if msg.status_list:
            latest_status = msg.status_list[-1]
            self.current_goal_status = latest_status.status
            
            if latest_status.status == 3:  # SUCCEEDED
                rospy.loginfo("✅ Goal reached successfully!")
                self.goal_reached = True
                self.stuck_counter = 0
                self.recovery_mode = False
                self.consecutive_same_goals = 0
            elif latest_status.status in [4, 5]:  # ABORTED or REJECTED
                rospy.logwarn(f"❌ Goal failed with status: {latest_status.status}")
                self.handle_failed_goal()
    
    def handle_failed_goal(self):
        """Handle a failed navigation goal with proper blacklisting"""
        self.goal_reached = True
        self.stuck_counter += 1
        
        # Add current goal to blacklist
        if self.current_goal:
            goal_key = self.goal_to_key(self.current_goal)
            
            # Increment attempt count
            self.goal_attempt_count[goal_key] = self.goal_attempt_count.get(goal_key, 0) + 1
            
            # Add to failed goals if max attempts reached
            if self.goal_attempt_count[goal_key] >= self.max_attempts_per_goal:
                self.failed_goals.add(goal_key)
                rospy.logwarn(f"🚫 Blacklisting goal {self.current_goal} after {self.goal_attempt_count[goal_key]} attempts")
            
            # Clean up old failed goals if too many
            if len(self.failed_goals) > 20:
                old_goals = list(self.failed_goals)[:5]
                for old_goal in old_goals:
                    self.failed_goals.remove(old_goal)
                rospy.loginfo(f"🧹 Cleaned up {len(old_goals)} old failed goals")
        
        rospy.logwarn(f"Goal failed! Stuck counter: {self.stuck_counter}/{self.max_stuck_count}")
        rospy.logwarn(f"Total blacklisted goals: {len(self.failed_goals)}")
    
    def goal_to_key(self, goal_coords, precision=1):
        """Convert goal coordinates to a hashable key with precision"""
        if goal_coords is None:
            return None
        x, y = goal_coords
        return (round(x * precision) / precision, round(y * precision) / precision)
    
    def is_goal_blacklisted(self, goal_coords):
        """Check if a goal is too close to any blacklisted goal"""
        if not goal_coords:
            return False
        
        goal_key = self.goal_to_key(goal_coords)
        if goal_key in self.failed_goals:
            return True
        
        # Also check if it's within blacklist radius of any failed goal
        gx, gy = goal_coords
        for failed_key in self.failed_goals:
            fx, fy = failed_key
            distance = math.sqrt((gx - fx)**2 + (gy - fy)**2)
            if distance < self.blacklist_radius:
                return True
        
        return False
    
    def cycle_detection(self, event):
        """Detect if robot is stuck in repetitive goal cycles"""
        if self.last_goal_coords == self.current_goal:
            self.consecutive_same_goals += 1
        else:
            self.consecutive_same_goals = 0
            self.last_goal_coords = self.current_goal
        
        if self.consecutive_same_goals >= 3:
            rospy.logwarn(f"🔄 CYCLE DETECTED! Same goal {self.consecutive_same_goals} times: {self.current_goal}")
            # Force blacklist this goal immediately
            if self.current_goal:
                goal_key = self.goal_to_key(self.current_goal)
                self.failed_goals.add(goal_key)
                rospy.logwarn(f"🚫 Emergency blacklisting: {goal_key}")
            
            # Force recovery
            self.stuck_counter = self.max_stuck_count
            self.consecutive_same_goals = 0
    
    def position_check(self, event):
        """Enhanced position-based stuck detection"""
        if len(self.position_history) < 5:
            return
        
        # Get positions from last 7.5 seconds (5 positions at 1.5s intervals)
        recent_positions = list(self.position_history)[-5:]
        
        # Calculate total movement
        total_distance = 0
        for i in range(1, len(recent_positions)):
            pos1, _ = recent_positions[i-1]
            pos2, _ = recent_positions[i]
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        # Check if robot is barely moving and has an active goal
        if total_distance < self.min_movement_threshold and not self.goal_reached:
            rospy.logwarn(f"🐌 Position-based stuck detection: {total_distance:.4f}m in 7.5s")
            self.stuck_counter += 1
            self.goal_reached = True  # Force new goal
    
    def analyze_map(self):
        """Analyze map and activate exploration"""
        if self.map_data is None:
            return
        
        # Count cell types
        unknown_cells = np.sum(self.map_data == -1)
        free_cells = np.sum(self.map_data == 0)
        occupied_cells = np.sum(self.map_data > 0)
        
        total_cells = self.map_data.size
        known_cells = free_cells + occupied_cells
        exploration_percentage = (known_cells / total_cells) * 100
        
        rospy.loginfo_throttle(15, f"📊 Map: {exploration_percentage:.1f}% explored ({known_cells}/{total_cells})")
        
        # Find frontiers
        frontiers = self.find_frontiers()
        valid_frontiers = self.filter_frontiers(frontiers)
        
        rospy.loginfo_throttle(15, f"   Frontiers: {len(frontiers)} total, {len(valid_frontiers)} valid")
        rospy.loginfo_throttle(15, f"   Blacklisted: {len(self.failed_goals)} goals")
        
        # Activate exploration with lower threshold
        if known_cells > 200 and len(valid_frontiers) > 0:
            if not self.exploration_active:
                rospy.loginfo("🚀 Exploration activated!")
                self.exploration_active = True
    
    def find_frontiers(self):
        """Find frontier cells efficiently"""
        if self.map_data is None:
            return []
        
        frontiers = []
        height, width = self.map_data.shape
        
        # Sample every 2nd cell for speed
        for y in range(3, height - 3, 2):
            for x in range(3, width - 3, 2):
                if self.map_data[y, x] == 0:  # Free cell
                    # Check for adjacent unknown cells
                    has_unknown = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if self.map_data[y + dy, x + dx] == -1:
                                has_unknown = True
                                break
                        if has_unknown:
                            break
                    
                    if has_unknown:
                        # Convert to world coordinates
                        world_x = x * self.map_info.resolution + self.map_info.origin.position.x
                        world_y = y * self.map_info.resolution + self.map_info.origin.position.y
                        frontiers.append((world_x, world_y))
        
        return frontiers
    
    def filter_frontiers(self, frontiers):
        """Filter frontiers with blacklisting"""
        if not frontiers or self.current_pose is None:
            return []
        
        valid_frontiers = []
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        for frontier in frontiers:
            fx, fy = frontier
            distance = math.sqrt((robot_x - fx)**2 + (robot_y - fy)**2)
            
            # Basic distance filtering
            if distance < 0.8 or distance > 10.0:
                continue
            
            # Check blacklist
            if self.is_goal_blacklisted(frontier):
                continue
            
            valid_frontiers.append((frontier, distance))
        
        # Sort by distance
        valid_frontiers.sort(key=lambda x: x[1])
        return [f[0] for f in valid_frontiers]
    
    def select_frontier(self, valid_frontiers):
        """Select frontier using different strategies to avoid repetition"""
        if not valid_frontiers:
            return None
        
        if self.frontier_selection_strategy == "round_robin":
            # Round-robin selection to avoid same frontier
            self.last_selected_frontier_index = (self.last_selected_frontier_index + 1) % len(valid_frontiers)
            return valid_frontiers[self.last_selected_frontier_index]
        
        elif self.frontier_selection_strategy == "random":
            return random.choice(valid_frontiers)
        
        else:  # distance
            return valid_frontiers[0]  # Closest
    
    def exploration_control(self, event):
        """Main exploration control with cycle breaking"""
        current_time = rospy.Time.now().to_sec()
        
        if not self.exploration_active:
            rospy.loginfo_throttle(20, "⏳ Waiting for map data...")
            return
        
        if self.current_pose is None:
            rospy.loginfo_throttle(10, "📍 Waiting for pose...")
            return
        
        # Handle stuck robot
        if self.stuck_counter >= self.max_stuck_count:
            rospy.logwarn("🔄 Initiating recovery - robot stuck!")
            self.advanced_recovery()
            return
        
        # Check for new goal
        time_since_goal = current_time - self.last_goal_time
        if self.goal_reached or time_since_goal > self.goal_timeout:
            if time_since_goal > self.goal_timeout:
                rospy.logwarn(f"⏰ Goal timeout ({self.goal_timeout}s)")
                # Blacklist timed-out goal
                if self.current_goal:
                    goal_key = self.goal_to_key(self.current_goal)
                    self.failed_goals.add(goal_key)
            
            self.send_exploration_goal()
    
    def advanced_recovery(self):
        """Multi-stage recovery process"""
        rospy.loginfo("🆘 Advanced recovery sequence starting...")
        self.recovery_mode = True
        
        # Stop robot
        self.stop_robot()
        rospy.sleep(1.0)
        
        # Choose recovery strategy based on stuck counter
        recovery_type = self.stuck_counter % 4
        
        if recovery_type == 0:
            rospy.loginfo("Recovery: Long backward movement")
            self.move_backward(0.3, 3.0)  # 0.3 m/s for 3 seconds
        elif recovery_type == 1:
            rospy.loginfo("Recovery: Random rotation and forward")
            angle = random.uniform(-math.pi, math.pi)
            self.rotate_to_angle(angle)
            rospy.sleep(0.5)
            self.move_forward(0.2, 2.0)
        elif recovery_type == 2:
            rospy.loginfo("Recovery: Spiral movement")
            self.spiral_movement()
        else:
            rospy.loginfo("Recovery: Large-angle rotation")
            self.rotate_robot(random.choice([-1.5, 1.5]), 2.0)
        
        # Clear some blacklisted goals if too many
        if len(self.failed_goals) > 15:
            old_goals = list(self.failed_goals)[:5]
            for goal in old_goals:
                self.failed_goals.remove(goal)
            rospy.loginfo(f"🧹 Cleared {len(old_goals)} blacklisted goals during recovery")
        
        # Reset states
        self.stuck_counter = 0
        self.goal_reached = True
        self.recovery_mode = False
        self.consecutive_same_goals = 0
        
        rospy.sleep(2.0)  # Wait before next goal
    
    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
    
    def move_backward(self, speed, duration):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -speed
        
        for _ in range(int(duration * 10)):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        self.stop_robot()
    
    def move_forward(self, speed, duration):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = speed
        
        for _ in range(int(duration * 10)):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        self.stop_robot()
    
    def rotate_to_angle(self, target_angle):
        """Rotate to specific angle"""
        if self.current_pose is None:
            return
        
        # Get current orientation
        orientation = self.current_pose.orientation
        current_angle = math.atan2(2.0 * (orientation.w * orientation.z), 
                                 1.0 - 2.0 * (orientation.z * orientation.z))
        
        angle_diff = target_angle - current_angle
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Rotate
        angular_speed = 1.0 if angle_diff > 0 else -1.0
        duration = abs(angle_diff) / abs(angular_speed)
        
        self.rotate_robot(angular_speed, duration)
    
    def rotate_robot(self, angular_vel, duration):
        """Rotate robot"""
        cmd = Twist()
        cmd.angular.z = angular_vel
        
        for _ in range(int(duration * 10)):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        self.stop_robot()
    
    def spiral_movement(self):
        """Spiral movement pattern"""
        cmd = Twist()
        
        for i in range(30):  # 3 second spiral
            cmd.linear.x = 0.1 + i * 0.01  # Increasing speed
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        self.stop_robot()
    
    def send_exploration_goal(self):
        """Send goal with anti-repetition logic"""
        frontiers = self.find_frontiers()
        valid_frontiers = self.filter_frontiers(frontiers)
        
        if not valid_frontiers:
            rospy.loginfo("🎉 No valid frontiers - exploration complete or need recovery!")
            if len(self.failed_goals) > 5:
                # Clear half the blacklist and try again
                failed_list = list(self.failed_goals)
                clear_count = len(failed_list) // 2
                for i in range(clear_count):
                    self.failed_goals.remove(failed_list[i])
                rospy.loginfo(f"🧹 Cleared {clear_count} blacklisted goals, retrying...")
                return
            else:
                # No frontiers and few blacklisted - probably done
                return
        
        # Select frontier using strategy
        selected_frontier = self.select_frontier(valid_frontiers)
        if not selected_frontier:
            return
        
        target_x, target_y = selected_frontier
        
        # Check if this is the same as last goal (emergency check)
        if self.current_goal and abs(target_x - self.current_goal[0]) < 0.1 and abs(target_y - self.current_goal[1]) < 0.1:
            rospy.logwarn("🚨 Same goal detected! Switching to random selection")
            self.frontier_selection_strategy = "random"
            selected_frontier = self.select_frontier(valid_frontiers)
            target_x, target_y = selected_frontier
        
        # Create and send goal
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = target_x
        goal.pose.position.y = target_y
        goal.pose.position.z = 0.0
        
        # Set orientation
        if self.current_pose is not None:
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y
            angle = math.atan2(target_y - robot_y, target_x - robot_x)
            goal.pose.orientation.z = math.sin(angle / 2)
            goal.pose.orientation.w = math.cos(angle / 2)
        else:
            goal.pose.orientation.w = 1.0
        
        # Update state and publish
        self.current_goal = (target_x, target_y)
        self.goal_pub.publish(goal)
        self.last_goal_time = rospy.Time.now().to_sec()
        self.goal_reached = False
        
        distance = math.sqrt((self.current_pose.position.x - target_x)**2 + 
                           (self.current_pose.position.y - target_y)**2)
        
        rospy.loginfo(f"🎯 NEW GOAL: ({target_x:.2f}, {target_y:.2f}) | Dist: {distance:.2f}m")
        rospy.loginfo(f"   Strategy: {self.frontier_selection_strategy} | Available: {len(valid_frontiers)} | Blacklisted: {len(self.failed_goals)}")

if __name__ == "__main__":
    try:
        VLMObjectDetectionExplorer()
    except rospy.ROSInterruptException:
        pass