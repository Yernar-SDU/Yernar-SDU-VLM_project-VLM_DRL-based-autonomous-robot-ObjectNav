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
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading
# Import the VLM detector #test
from pixel_to_cords import VLM_Response_Processor


class RobotExplorer:
    
    
    def __init__(self, vlmType = 'moondream', world = 'TD3.world'):
        self.vlm_type = vlmType
        # Launch system
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
        print("LiDAR system launched!")
        time.sleep(4.0)
        
        rospy.init_node('robot_explorer', anonymous=False)
        
        # Parameters
        self.camera_topic = "/realsense_camera/color/image_raw"
        self.pointcloud_topic = "/realsense_camera/depth/color/points"
        # self.camera_info_topic = "/realsense_camera/color/camera_info"
        self.map_topic = "/map"
        self.odom_topic = "/r1/odom"
        self.save_dir = "/tmp/exploration_data"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Initialize VLM detector
        self.vlm_detector = VLM_Response_Processor(self.vlm_type, save_dir=self.save_dir)
        
        # Timing controls
        self.last_image_save_time = rospy.Time.now().to_sec()
        self.last_object_detection_time = rospy.Time.now().to_sec()
        self.object_detection_interval = 2 # seconds
       
        # File paths
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        
        # Create metadata file if it doesn't exist
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
        self.stop_condition = False

        # Publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        
        # Subscribers
        # rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        
        # Synchronized subscribers
        image_sub = Subscriber(self.camera_topic, Image)
        pc_sub = Subscriber(self.pointcloud_topic, PointCloud2)
        odom_sub = Subscriber(self.odom_topic, Odometry)
        sync = ApproximateTimeSynchronizer(
            [image_sub, pc_sub, odom_sub], 
            queue_size=100, 
            slop=0.01
        )
        sync.registerCallback(self.synchronized_callback)
        
        rospy.loginfo("="*60)
        rospy.loginfo("ROBOT EXPLORER WITH VLM OBJECT DETECTION")
        rospy.loginfo("="*60)
        rospy.loginfo("Features: Autonomous exploration + VLM object detection")
        
        # Main control loops
        # self.exploration_control()
        rospy.Timer(rospy.Duration(1.5), self.position_check)
        rospy.Timer(rospy.Duration(3.0), self.cycle_detection)
        rospy.Timer(rospy.Duration(0.5), self.exploration_control)
        rospy.spin()

    def synchronized_callback(self, image_msg, pc_msg, odom_msg):
        """Handle synchronized sensor data"""
        try:
            current_time = rospy.Time.now().to_sec()

            # Object detection with VLM every N seconds
            if int(current_time) % int(self.object_detection_interval) == 0:
                if current_time - self.last_object_detection_time >= self.object_detection_interval:
                    
                    rospy.loginfo("⏸️ Pausing exploration for VLM detection...")
                    
                    # Cancel current navigation goal to stop robot
                    # self.cancel_current_goal()
                    
                    # Capture synchronized data
                    
                    # Run detection using VLM detector
                    print('BEFOREBEFOREBEFOREBEFOREBEFOREBEFOREBEFORE', odom_msg.pose.pose)
                    # num_detected = self.vlm_detector.detect_objects(
                    #     cv_image, 
                    #     pc_msg,
                    #     odom_msg.pose.pose,
                    #     current_time,
                    #     self.frame_id
                    # )
                    current_time_start = rospy.Time.now().to_sec()
                    start_ts = rospy.Time.now().to_sec()
                    def processing_wrapper(img, pc, pose, ts, start_time):
                        # This runs in background
                        num_detected = self.vlm_detector.detect_objects(img, pc, pose, ts, self.frame_id)
                        
                        # Calculate true latency
                        diff = rospy.Time.now().to_sec() - start_time
                        with open("latencies.txt", "a") as f:
                            f.write(f"Time: {ts}, Latency: {diff:.4f}s, Count: {num_detected}\n")
                        
                        rospy.loginfo(f"✅ VLM Finished. Detected: {num_detected} in {diff:.2f}s")
                        self.save_image_with_metadata(img, current_time, pose)
                        rospy.loginfo(f"▶️ Resuming exploration (detected {num_detected} objects)")
                    
                    
                    cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                    # Start one single thread
                    threading.Thread(target=processing_wrapper, 
                                    args=(cv_image, pc_msg, odom_msg.pose.pose, current_time, start_ts)).start()
                    
                    print('AFTERAFTERAFTERAFTERAFTERAFTERAFTERAFTERAFTERAFTER', odom_msg.pose.pose)
                    current_time_finish = rospy.Time.now().to_sec()
                    # Save image with metadata

                    # Resume exploration
                    self.goal_reached = True
                    self.last_object_detection_time = current_time
                    


                    diff = current_time_finish - current_time_start
                    # TODO just save diff's variable
                    with open("latencies.txt", "a") as f:
                        f.write(f"{current_time}, {diff}\n")




        except Exception as e:
            rospy.logerr(f"Synchronized callback error: {e}")

    def cancel_current_goal(self):
        """Cancel the current navigation goal"""
        rospy.sleep(0.1)
        self.cancel_pub.publish(GoalID())
        rospy.loginfo("Navigation goal cancelled")
    
    def save_image_with_metadata(self, cv_image, current_time, current_pose):
        """Save image with metadata"""
        # filename = os.path.join(self.save_dir, f"frame_{self.frame_id:05d}.png")
        # cv2.imwrite(filename, cv_image)
        
        # Save metadata
        metadata = {
            # "id": f"img_{self.frame_id:06d}",
            "timestamp": current_time,
            # "image_path": filename,
        }
        
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
                "map_cell_y": map_cell_y
            })
        else:
            rospy.logwarn_throttle(10, "Saving image without pose/map metadata")
        
        # Append to JSONL
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")
        
        self.frame_id += 1
        rospy.loginfo(f"Saved frame {self.frame_id} at {current_time:.1f}s")
    
    def map_callback(self, msg):
        """Handle map updates"""
        self.current_map = msg
        self.map_info = msg.info
        
        # Convert map data to numpy array
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        if self.map_data is None or self.map_data.size == 0:
            rospy.logwarn("Received empty or invalid map data")
            return 
        
        # Save map image
        grid_img = np.zeros_like(self.map_data, dtype=np.uint8)
        grid_img[self.map_data == -1] = 127  # Unknown = gray
        grid_img[self.map_data == 0] = 255   # Free = white
        grid_img[self.map_data > 0] = 0      # Occupied = black
        
        map_filename = os.path.join(self.save_dir, "map_latest.png")
        cv2.imwrite(map_filename, grid_img)
        
        self.analyze_map()
    
    def odom_callback(self, msg):
        """Handle odometry updates"""
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
        """Convert goal coordinates to a hashable key"""
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
        
        # Also check if it's within blacklist radius
        gx, gy = goal_coords
        for failed_key in self.failed_goals:
            fx, fy = failed_key
            distance = math.sqrt((gx - fx)**2 + (gy - fy)**2)
            if distance < self.blacklist_radius:
                return True
        
        return False
    
    def cycle_detection(self, event):
        """Detect if robot is stuck in repetitive goal cycles"""
        if self.stop_condition:
            return
        
        if self.last_goal_coords == self.current_goal:
            self.consecutive_same_goals += 1
        else:
            self.consecutive_same_goals = 0
            self.last_goal_coords = self.current_goal
        
        if self.consecutive_same_goals >= 3:
            rospy.logwarn(f"🔄 CYCLE DETECTED! Same goal {self.consecutive_same_goals} times")
            if self.current_goal:
                goal_key = self.goal_to_key(self.current_goal)
                self.failed_goals.add(goal_key)
                rospy.logwarn(f"🚫 Emergency blacklisting: {goal_key}")
            
            self.stuck_counter = self.max_stuck_count
            self.consecutive_same_goals = 0
    
    def position_check(self, event):
        """Position-based stuck detection"""
        if self.stop_condition:
            return
        
        if len(self.position_history) < 5:
            return
        
        recent_positions = list(self.position_history)[-5:]
        
        total_distance = 0
        for i in range(1, len(recent_positions)):
            pos1, _ = recent_positions[i-1]
            pos2, _ = recent_positions[i]
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        if total_distance < self.min_movement_threshold and not self.goal_reached:
            rospy.logwarn(f"🐌 Position-based stuck detection: {total_distance:.4f}m in 7.5s")
            self.stuck_counter += 1
            self.goal_reached = True
    
    def analyze_map(self):
        """Analyze map and activate exploration"""
        if self.map_data is None:
            return
        
        unknown_cells = np.sum(self.map_data == -1)
        free_cells = np.sum(self.map_data == 0)
        occupied_cells = np.sum(self.map_data > 0)
        
        total_cells = self.map_data.size
        known_cells = free_cells + occupied_cells
        exploration_percentage = (known_cells / total_cells) * 100
        
        rospy.loginfo_throttle(15, f"📊 Map: {exploration_percentage:.1f}% explored ({known_cells}/{total_cells})")
        
        frontiers = self.find_frontiers()
        valid_frontiers = self.filter_frontiers(frontiers)
        
        rospy.loginfo_throttle(15, f"   Frontiers: {len(frontiers)} total, {len(valid_frontiers)} valid")
        rospy.loginfo_throttle(15, f"   Blacklisted: {len(self.failed_goals)} goals")
        
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
        
        for y in range(3, height - 3, 2):
            for x in range(3, width - 3, 2):
                if self.map_data[y, x] == 0:
                    has_unknown = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if self.map_data[y + dy, x + dx] == -1:
                                has_unknown = True
                                break
                        if has_unknown:
                            break
                    
                    if has_unknown:
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
            
            if distance < 0.8 or distance > 10.0:
                continue
            
            if self.is_goal_blacklisted(frontier):
                continue
            
            valid_frontiers.append((frontier, distance))
        
        valid_frontiers.sort(key=lambda x: x[1])
        return [f[0] for f in valid_frontiers]
    
    def select_frontier(self, valid_frontiers):
        """Select frontier using different strategies"""
        if not valid_frontiers:
            return None
        
        if self.frontier_selection_strategy == "round_robin":
            self.last_selected_frontier_index = (self.last_selected_frontier_index + 1) % len(valid_frontiers)
            return valid_frontiers[self.last_selected_frontier_index]
        elif self.frontier_selection_strategy == "random":
            return random.choice(valid_frontiers)
        else:
            return valid_frontiers[0]
    
    def exploration_control(self, event):
        """Main exploration control loop"""
        if self.stop_condition:
            return

        current_time = rospy.Time.now().to_sec()
        
        if not self.exploration_active:
            rospy.loginfo_throttle(20, "⏳ Waiting for map data...")
            return
        
        if self.current_pose is None:
            rospy.loginfo_throttle(10, "📍 Waiting for pose...")
            return
        
        if self.stuck_counter >= self.max_stuck_count:
            rospy.logwarn("🔄 Initiating recovery - robot stuck!")
            self.advanced_recovery()
            return
        
        time_since_goal = current_time - self.last_goal_time
        if self.goal_reached or time_since_goal > self.goal_timeout:
            if time_since_goal > self.goal_timeout:
                rospy.logwarn(f"⏰ Goal timeout ({self.goal_timeout}s)")
                if self.current_goal:
                    goal_key = self.goal_to_key(self.current_goal)
                    self.failed_goals.add(goal_key)
            
            self.send_exploration_goal()
    
    def advanced_recovery(self):
        """Multi-stage recovery process"""
        rospy.loginfo("🆘 Advanced recovery sequence starting...")
        self.recovery_mode = True
        
        self.stop_robot()
        rospy.sleep(1.0)
        
        recovery_type = self.stuck_counter % 4
        
        if recovery_type == 0:
            rospy.loginfo("Recovery: Long backward movement")
            self.move_backward(0.3, 3.0)
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
        
        if len(self.failed_goals) > 15:
            old_goals = list(self.failed_goals)[:5]
            for goal in old_goals:
                self.failed_goals.remove(goal)
            rospy.loginfo(f"🧹 Cleared {len(old_goals)} blacklisted goals during recovery")
        
        self.stuck_counter = 0
        self.goal_reached = True
        self.recovery_mode = False
        self.consecutive_same_goals = 0
        
        rospy.sleep(2.0)
    
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
        
        orientation = self.current_pose.orientation
        current_angle = math.atan2(2.0 * (orientation.w * orientation.z), 
                                   1.0 - 2.0 * (orientation.z * orientation.z))
        
        angle_diff = target_angle - current_angle
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
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
        for i in range(30):
            cmd.linear.x = 0.1 + i * 0.01
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
                failed_list = list(self.failed_goals)
                clear_count = len(failed_list) // 2
                for i in range(clear_count):
                    self.failed_goals.remove(failed_list[i])
                rospy.loginfo(f"🧹 Cleared {clear_count} blacklisted goals, retrying...")
                return
            else:
                return
        
        selected_frontier = self.select_frontier(valid_frontiers)
        if not selected_frontier:
            return
        
        target_x, target_y = selected_frontier
        
        if self.current_goal and abs(target_x - self.current_goal[0]) < 0.1 and abs(target_y - self.current_goal[1]) < 0.1:
            rospy.logwarn("🚨 Same goal detected! Switching to random selection")
            self.frontier_selection_strategy = "random"
            selected_frontier = self.select_frontier(valid_frontiers)
            target_x, target_y = selected_frontier
        
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = target_x
        goal.pose.position.y = target_y
        goal.pose.position.z = 0.0
        
        if self.current_pose is not None:
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y
            angle = math.atan2(target_y - robot_y, target_x - robot_x)
            goal.pose.orientation.z = math.sin(angle / 2)
            goal.pose.orientation.w = math.cos(angle / 2)
        else:
            goal.pose.orientation.w = 1.0
        
        self.current_goal = (target_x, target_y)
        self.goal_pub.publish(goal)
        self.last_goal_time = rospy.Time.now().to_sec()
        self.goal_reached = False
        
        distance = math.sqrt((self.current_pose.position.x - target_x)**2 + 
                           (self.current_pose.position.y - target_y)**2)
        
        rospy.loginfo(f"🎯 NEW GOAL: ({target_x:.2f}, {target_y:.2f}) | Dist: {distance:.2f}m")
        rospy.loginfo(f"   Strategy: {self.frontier_selection_strategy} | Available: {len(valid_frontiers)} | Blacklisted: {len(self.failed_goals)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOTA Navigation Runner")
    parser.add_argument("--vlm", "-a", type=str, default="moondream",
                       choices=["moondream", "gpt4o"],
                       help="VLM algorithm")
    parser.add_argument("--world", "-w", type=str, default="TD3.world",
                       help="Gazebo world file (e.g. TD3.world, turtlebot3_house.world)")
    args = parser.parse_args()


    try:
        RobotExplorer(args.vlm, args.world)
    except rospy.ROSInterruptException:
        pass