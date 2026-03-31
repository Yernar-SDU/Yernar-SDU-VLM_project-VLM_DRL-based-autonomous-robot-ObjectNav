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
# Import the VLM detector
from pixel_to_cords import VLM_Response_Processor


class RobotExplorer:


    def __init__(self, vlmType='moondream', world='TD3_signs2.world'):
        self.vlm_type = vlmType
        self.permanent_blacklist = set()      # Bug C: survives recovery clears
        self.permanent_blacklist_threshold = 8  # failures before permanent ban
        self.last_goal_send_time = 0          # Bug B: rate limiter
        self.min_goal_send_interval = 4.0     # Bug B: seconds between sends
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
        self.object_detection_interval = 2  # seconds

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
        self.max_stuck_count = 5

        # Position tracking for stuck detection
        # FIX 5: Larger history window — 50 entries @ 1.5s = 75s of history
        self.position_history = deque(maxlen=5000)
        self.last_position_check = 0

        # FIX 5: Raised threshold so slow obstacle navigation doesn't falsely trigger
        self.min_movement_threshold = 0.3

        # Goal management with proper blacklisting
        self.current_goal = None
        self.failed_goals = set()
        self.goal_attempt_count = {}
        # FIX 3: More attempts before permanent blacklisting
        self.max_attempts_per_goal = 6

        # FIX 3: Smaller blacklist radius so a failed goal doesn't poison a large area
        self.blacklist_radius = 0.4

        self.last_selected_frontier_index = 0
        self.frontier_selection_strategy = "closest"

        # Exploration state
        self.exploration_active = False
        self.last_goal_time = 0

        # FIX 6: Longer timeout — cluttered small rooms need more navigation time
        self.goal_timeout = 50.0

        self.recovery_mode = False
        self.consecutive_same_goals = 0
        self.last_goal_coords = None
        self.stop_condition = False
        self.visited_positions = deque(maxlen=20)

        # Publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)

        # Subscribers
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

        rospy.loginfo("=" * 60)
        rospy.loginfo("ROBOT EXPLORER WITH VLM OBJECT DETECTION")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Features: Autonomous exploration + VLM object detection")

        rospy.Timer(rospy.Duration(1.5), self.position_check)
        rospy.Timer(rospy.Duration(3.0), self.cycle_detection)
        rospy.Timer(rospy.Duration(0.5), self.exploration_control)
        rospy.spin()

    def synchronized_callback(self, image_msg, pc_msg, odom_msg):
        """Handle synchronized sensor data"""
        try:
            current_time = rospy.Time.now().to_sec()

            if int(current_time) % int(self.object_detection_interval) == 0:
                if current_time - self.last_object_detection_time >= self.object_detection_interval:

                    rospy.loginfo("⏸️ Pausing exploration for VLM detection...")

                    current_time_start = rospy.Time.now().to_sec()
                    start_ts = rospy.Time.now().to_sec()

                    def processing_wrapper(img, pc, pose, ts, start_time):
                        num_detected = self.vlm_detector.detect_objects(img, pc, pose, ts, self.frame_id)
                        diff = rospy.Time.now().to_sec() - start_time
                        with open("latencies.txt", "a") as f:
                            f.write(f"Time: {ts}, Latency: {diff:.4f}s, Count: {num_detected}\n")
                        rospy.loginfo(f"✅ VLM Finished. Detected: {num_detected} in {diff:.2f}s")
                        self.save_image_with_metadata(img, current_time, pose)
                        rospy.loginfo(f"▶️ Resuming exploration (detected {num_detected} objects)")

                    cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                    threading.Thread(
                        target=processing_wrapper,
                        args=(cv_image, pc_msg, odom_msg.pose.pose, current_time, start_ts)
                    ).start()

                    current_time_finish = rospy.Time.now().to_sec()
                    self.goal_reached = True
                    self.last_object_detection_time = current_time

                    diff = current_time_finish - current_time_start
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
        metadata = {
            "timestamp": current_time,
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

        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        self.frame_id += 1
        rospy.loginfo(f"Saved frame {self.frame_id} at {current_time:.1f}s")

    def map_callback(self, msg):
        """Handle map updates"""
        self.current_map = msg
        self.map_info = msg.info

        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        if self.map_data is None or self.map_data.size == 0:
            rospy.logwarn("Received empty or invalid map data")
            return

        grid_img = np.zeros_like(self.map_data, dtype=np.uint8)
        grid_img[self.map_data == -1] = 127  # Unknown = gray
        grid_img[self.map_data == 0] = 255   # Free = white
        grid_img[self.map_data > 0] = 0      # Occupied = black

        map_filename = os.path.join(self.save_dir, "map_latest.png")
        cv2.imwrite(map_filename, np.flipud(grid_img))

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        current_pos = (self.current_pose.position.x, self.current_pose.position.y)
        self.position_history.append((current_pos, rospy.Time.now().to_sec()))

        if not self.visited_positions or math.sqrt(
            (current_pos[0] - self.visited_positions[-1][0]) ** 2 +
            (current_pos[1] - self.visited_positions[-1][1]) ** 2
        ) > 0.5:
            self.visited_positions.append(current_pos)

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
        """Handle a failed navigation goal."""
        # Bug B fix: don't set goal_reached=True immediately — let the rate
        # limiter in exploration_control control when the next goal fires.
        self.stuck_counter += 1

        if self.current_goal:
            goal_key = self.goal_to_key(self.current_goal)
            self.goal_attempt_count[goal_key] = self.goal_attempt_count.get(goal_key, 0) + 1
            count = self.goal_attempt_count[goal_key]

            # Bug C fix: permanent blacklist for truly unreachable goals
            if count >= self.permanent_blacklist_threshold:
                self.permanent_blacklist.add(goal_key)
                rospy.logwarn(f"🔒 PERMANENT blacklist: {goal_key} after {count} attempts")
            elif count >= self.max_attempts_per_goal:
                self.failed_goals.add(goal_key)
                rospy.logwarn(f"🚫 Blacklisting goal {self.current_goal} after {count} attempts")

            if len(self.failed_goals) > 10:
                old_goals = list(self.failed_goals)[:5]
                for old_goal in old_goals:
                    self.failed_goals.remove(old_goal)

        # Bug B fix: mark goal as reached so control loop can proceed,
        # but the rate limiter will prevent immediate re-send
        self.goal_reached = True

        rospy.logwarn(f"Goal failed! Stuck counter: {self.stuck_counter}/{self.max_stuck_count}")
        rospy.logwarn(f"Blacklisted: {len(self.failed_goals)} temp, {len(self.permanent_blacklist)} permanent")

    def goal_to_key(self, goal_coords, precision=1):
        """Convert goal coordinates to a hashable key"""
        if goal_coords is None:
            return None
        x, y = goal_coords
        return (round(x * precision) / precision, round(y * precision) / precision)

    def is_goal_blacklisted(self, goal_coords):
        """Check temp AND permanent blacklists."""
        if not goal_coords:
            return False

        goal_key = self.goal_to_key(goal_coords)

        # Bug C fix: check permanent blacklist first — these never clear
        if goal_key in self.permanent_blacklist:
            return True

        if goal_key in self.failed_goals:
            return True

        gx, gy = goal_coords
        for failed_key in list(self.failed_goals) + list(self.permanent_blacklist):
            fx, fy = failed_key
            if math.sqrt((gx - fx) ** 2 + (gy - fy) ** 2) < self.blacklist_radius:
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

        # FIX 5: Need more history entries before triggering (10 instead of 5)
        if len(self.position_history) < 10:
            return

        # FIX 5: Use last 10 entries (~15s) for a more reliable movement estimate
        recent_positions = list(self.position_history)[-10:]

        total_distance = 0
        for i in range(1, len(recent_positions)):
            pos1, _ = recent_positions[i - 1]
            pos2, _ = recent_positions[i]
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            total_distance += math.sqrt(dx * dx + dy * dy)

        # FIX 5: Lowered threshold (0.3m) so legitimate slow navigation doesn't falsely trigger
        if total_distance < self.min_movement_threshold and not self.goal_reached:
            rospy.logwarn(f"🐌 Position-based stuck detection: {total_distance:.4f}m in ~15s")
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
        """Find frontier cells efficiently.

        FIX 4: Step size reduced from 2 to 1 so thin corridors and small
        unexplored pockets around obstacles are not missed.
        """
        if self.map_data is None:
            return []

        frontiers = []
        height, width = self.map_data.shape

        # FIX 4: step=1 instead of step=2 — full resolution scan
        for y in range(1, height - 1, 1):
            for x in range(1, width - 1, 1):
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
        if not frontiers or self.current_pose is None:
            return []

        valid_frontiers = []
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        for frontier in frontiers:
            fx, fy = frontier
            distance = math.sqrt((robot_x - fx) ** 2 + (robot_y - fy) ** 2)

            if distance < 0.1 or distance > 7.5:
                continue

            if self.is_goal_blacklisted(frontier):
                continue

            # FIX 2: Reduced visited exclusion radius from 0.8m → 0.35m so the
            # robot can revisit nearby areas in a small room without running out
            # of valid frontiers.  Also removed the "distance < 1.0" guard that
            # was doubly restricting close frontiers.
            already_visited = any(
                math.sqrt((fx - vx) ** 2 + (fy - vy) ** 2) < 0.35
                for vx, vy in self.visited_positions
            )
            if already_visited:
                continue

            valid_frontiers.append((frontier, distance))

        valid_frontiers.sort(key=lambda x: x[1])
        return [f[0] for f in valid_frontiers]

    def select_frontier(self, valid_frontiers):
        """Score all frontiers and pick the best one.

        FIX 1: The original code had a 30% random early-return *inside* the
        scoring loop, which meant the loop almost never evaluated more than 1-2
        frontiers.  The random fallback is now applied *after* the full loop so
        the scoring is always completed first.
        """
        if not valid_frontiers or not self.current_pose:
            return valid_frontiers[0] if valid_frontiers else None

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        q = self.current_pose.orientation
        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        best, best_score = None, -999

        for fx, fy in valid_frontiers:
            dist = math.sqrt((robot_x - fx) ** 2 + (robot_y - fy) ** 2)
            gain = self._count_unknown_near(fx, fy, radius=1.5)

            angle_to = math.atan2(fy - robot_y, fx - robot_x)
            angle_diff = abs(angle_to - robot_yaw)
            while angle_diff > math.pi:
                angle_diff = abs(angle_diff - 2 * math.pi)

            alignment = 1.0 - (angle_diff / math.pi)
            center_bias = dist

            score = (
                gain * 1.5 +
                alignment * 1.0 -
                dist * 0.2 +
                center_bias * 0.3
            )

            if score > best_score:
                best_score = score
                best = (fx, fy)

        # FIX 1: Random fallback applied AFTER the full scoring loop, not inside it.
        # 30% of the time we still allow a random pick to avoid getting stuck on
        # the same scored winner repeatedly.
        if random.random() < 0.3:
            return random.choice(valid_frontiers)

        return best

    def _count_unknown_near(self, world_x, world_y, radius=1.5):
        if self.map_data is None or self.map_info is None:
            return 0
        mx = int((world_x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((world_y - self.map_info.origin.position.y) / self.map_info.resolution)
        r = int(radius / self.map_info.resolution)
        h, w = self.map_data.shape
        x1, x2 = max(0, mx - r), min(w, mx + r)
        y1, y2 = max(0, my - r), min(h, my + r)
        return int(np.sum(self.map_data[y1:y2, x1:x2] == -1))

    def exploration_control(self, event):
        """Main exploration control loop"""
        if self.stop_condition:
            return
        self.analyze_map()

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
        """Multi-stage recovery — clears temp blacklist only, keeps permanent."""
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

        # Bug A+C fix: only clear TEMP blacklist — permanent stays forever
        cleared = len(self.failed_goals)
        self.failed_goals.clear()
        # Keep goal_attempt_count so counts accumulate toward permanent threshold
        rospy.loginfo(f"🧹 Cleared {cleared} temp blacklisted goals (permanent: {len(self.permanent_blacklist)})")

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

        angular_speed = 0.3 if angle_diff > 0 else -0.3
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
            cmd.angular.z = 0.3
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        self.stop_robot()

    def send_exploration_goal(self):
        """Send goal with rate limiting and permanent blacklist awareness."""
        # Bug B fix: hard rate limit — never send goals faster than min interval
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_goal_send_time < self.min_goal_send_interval:
            return

        frontiers = self.find_frontiers()
        valid_frontiers = self.filter_frontiers(frontiers)

        if not valid_frontiers:
            rospy.loginfo("🎉 No valid frontiers!")

            # Bug A fix: if permanent blacklist is eating all frontiers,
            # log it clearly — this likely means exploration is genuinely complete
            if len(self.permanent_blacklist) > 0:
                rospy.logwarn(f"⚠️ {len(self.permanent_blacklist)} permanently unreachable goals exist.")
                rospy.logwarn("Likely fully explored or map has unreachable phantom frontiers.")

            # Only clear TEMP blacklist, never permanent
            if len(self.failed_goals) > 0:
                cleared = len(self.failed_goals)
                self.failed_goals.clear()
                rospy.loginfo(f"🧹 Cleared {cleared} temp goals, retrying...")
            return

        selected_frontier = self.select_frontier(valid_frontiers)
        if not selected_frontier:
            return

        target_x, target_y = selected_frontier

        if (self.current_goal and
                abs(target_x - self.current_goal[0]) < 0.1 and
                abs(target_y - self.current_goal[1]) < 0.1):
            rospy.logwarn("🚨 Same goal again! Forcing random selection")
            selected_frontier = random.choice(valid_frontiers)
            target_x, target_y = selected_frontier

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = target_x
        goal.pose.position.y = target_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0

        self.current_goal = (target_x, target_y)
        self.goal_pub.publish(goal)
        self.last_goal_time = rospy.Time.now().to_sec()
        self.last_goal_send_time = current_time  # Bug B fix: update rate limiter
        self.goal_reached = False

        distance = math.sqrt(
            (self.current_pose.position.x - target_x) ** 2 +
            (self.current_pose.position.y - target_y) ** 2
        )
        rospy.loginfo(f"🎯 NEW GOAL: ({target_x:.2f}, {target_y:.2f}) | Dist: {distance:.2f}m")
        rospy.loginfo(f"   Available: {len(valid_frontiers)} | Temp BL: {len(self.failed_goals)} | Perm BL: {len(self.permanent_blacklist)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOTA Navigation Runner")
    parser.add_argument("--vlm", "-a", type=str, default="moondream",
                        choices=["moondream", "gpt4o"],
                        help="VLM algorithm")
    parser.add_argument("--world", "-w", type=str, default="TD3_signs2.world",
                        help="Gazebo world file (e.g. TD3_signs2.world, turtlebot3_house.world)")
    args = parser.parse_args()

    try:
        RobotExplorer(args.vlm, args.world)
    except rospy.ROSInterruptException:
        pass