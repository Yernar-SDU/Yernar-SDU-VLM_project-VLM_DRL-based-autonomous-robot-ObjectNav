#!/usr/bin/env python3
import math
import os
import random
import subprocess
import time
from os import path
from sensor_msgs.msg import Imu

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sensor_msgs.point_cloud2 as pc2
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import os
import glob
###############################################
#          Adjustable Parameters
###############################################
LAUNCHFILE = "multi_robot_scenario.launch"
ENVIRONMENT_DIM = 20
TIME_DELTA = 0.1
MAX_STEPS = 600
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.25

STUCK_STEPS = 70
STUCK_MOVEMENT_THRESHOLD = 0.02

NEAR_WALL_STEPS = 40
DISTANCE_SCALE = 0.01
BLUE_DISTANCE_THRESHOLD = 0.15

WALL_DISTANCE_THRESHOLD = 1
TIME_STEP_PENALTY = -1  # small negative reward each step

# For the depth image shape
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Gamma correction
GAMMA_VALUE = 0.8

###############################################
class GazeboEnv:
    """
    Environment returning:
      - Depth-based channel (1, 64, 64)
      - 7D array of scalars [prev_lin, prev_ang, last_lin, last_ang, dist2goal, angle2goal, min_laser].
    """

    def __init__(self):
        self.episode_num = 0
        self.environment_dim = ENVIRONMENT_DIM
        self.odom_x = 0
        self.odom_y = 0
        self.odom_yaw = 0.0

        self.goal_x = 1.96
        self.goal_y = 0.02

        self.ground_truth_y = 0
        self.ground_truth_x = 0

        self.episode_start_time = None
        self.episode_sim_time = 0.0  # Simulation time (steps * TIME_DELTA)
        self.total_time = 0
        # used for randomizing the environment
        self.upper = 5.0
        self.lower = -5.0

        # store the single normalized depth image
        self.normed_depth = None
        self.last_odom = None

        # For stuck detection
        self.stuck_counter = 0
        self.last_robot_x = None
        self.last_robot_y = None

        self.near_wall_counter = 0

        # Two-step action memory
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        # For progress reward
        self.prev_distance = None
        # For "wall" reward from the depth camera
        self.prev_wall_real = None
        self.min_wall_real = 10.0

        # New: Variables to track cumulative rotation
        self.cum_rotation = 0.0
        self.last_yaw = None

        # For setting model state
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"

        self.max_steps = 400
        self.current_step = 0
        # For IMU sensor
        # self.imu_sub = rospy.Subscriber("/imu/data", Imu, self.imu_callback, queue_size=1)
        # self.latest_imu_data = None # Variable to store the latest IMU reading
        
        # self.orientation_y = None
        # self.imu_data = None
            
        for proc in ["roscore", "rosmaster", "gzserver", "gzclient", "rtabmap"]:
            subprocess.run(["killall", "-9", proc], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(1)

        
        # ============================================
        # LAUNCH ROSCORE (QUIET)
        # ============================================
        port = "11311"
        subprocess.Popen(
            ["roscore", "-p", port],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Roscore launched!")
        time.sleep(2)
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03        
        # ============================================
        # INITIALIZE ROSPY
        # ============================================
        rospy.init_node("gym", anonymous=True)
        
        # ============================================
        # LAUNCH GAZEBO (QUIET)
        # ============================================
        if LAUNCHFILE.startswith("/"):
            fullpath = LAUNCHFILE
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", LAUNCHFILE)
        
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")
        
        subprocess.Popen(
            ["roslaunch", "-p", port, fullpath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Gazebo launched")
        time.sleep(4)
        
        subprocess.Popen([
            "roslaunch",
            "--log-level", "fatal",  # ← KEY: Suppress RTAB-Map verbosity
            "-p", port,
            "rtabmap_ros", "rtabmap.launch",
            "rtabmap_args:=--delete_db_on_start",
            "depth:=true",
            "rgb:=false",
            "scan:=false"
        ], 
        stdout=subprocess.DEVNULL,  # Suppress stdout
        stderr=subprocess.DEVNULL   # Suppress stderr
        )
        print("✅ RTAB-Map SLAM launched (silent)")
        time.sleep(2)


        # ROS publishers / services
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.publisher_goal = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher_lin = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher_ang = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber(
            "/realsense_camera/depth/image_raw", 
            Image, 
            self.depth_callback, 
            queue_size=1
        )
        self.odom_sub = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)
        # self.init_path_marker()
        rospy.sleep(3.0)

        # self.take_environment_screenshot()
        print("Environment ready!")

    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    # def take_environment_screenshot(self):
    #     """Take a clean screenshot of the environment without any markers."""
    #     os.makedirs(self.screenshot_dir, exist_ok=True)
        
    #     if self.overhead_camera_image is None:
    #         rospy.logwarn("No image from overhead camera yet.")
    #         return None
        
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(self.overhead_camera_image, "bgr8")
            
    #         # Save raw image - no drawings at all
    #         filename = os.path.join(self.screenshot_dir, 'environment.png')
    #         cv2.imwrite(filename, cv_image)
    #         rospy.loginfo(f"📸 Environment screenshot saved: {filename}")
    #         return filename
            
    #     except CvBridgeError as e:
    #         rospy.logerr(f"CV Bridge error: {e}")
    #         return None
    
    def odom_callback(self, od_data):
        self.last_odom = od_data
        self.odom_x = od_data.pose.pose.position.x
        self.odom_y = od_data.pose.pose.position.y
        # Convert orientation to yaw
        orientation_q = od_data.pose.pose.orientation
        quat = Quaternion(orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z)
        _, _, yaw = quat.to_euler()
        
        # Update cumulative rotation
        if self.last_yaw is not None:
            # Compute smallest angular difference
            delta_yaw = abs((yaw - self.last_yaw + math.pi) % (2 * math.pi) - math.pi)
            self.cum_rotation += delta_yaw
        self.last_yaw = yaw
        
        self.odom_yaw = yaw


        current_pos = (self.odom_x, self.odom_y)
    
       # Only add point if moved enough (avoid too many points)
        # if len(self.path_points_xy) == 0:
        #     self.path_points_xy.append(current_pos)
        # else:
        #     last = self.path_points_xy[-1]
        #     dist = math.sqrt((current_pos[0] - last[0])**2 + (current_pos[1] - last[1])**2)
        #     if dist > 0.05:  # Min 5cm between points
        #         self.path_points_xy.append(current_pos)

        # point = Point()
        # point.x = self.odom_x
        # point.y = self.odom_y
        # point.z = 0.1
        # self.path_points.append(point)
        # self.marker.points = self.path_points
        # self.marker.header.stamp = rospy.Time.now()
        # self.marker_pub.publish(self.marker)

    # def imu_callback(self, imu_data):
    #         """
    #         Callback function for the IMU subscriber.
    #         Stores the latest IMU data and prints it to the terminal. # <-- Modified description
    #         """
            # self.latest_imu_data = imu_data

            # --- ADD THIS LINE ---
            # Print the entire received message to the terminal
            # print("--- Received IMU Data: ---")
            # #print(imu_data)
            # print("--------------------------")
            # --- END ADDED LINES ---

            # Optional: You can still access specific fields if needed
            # orientation_q = imu_data.orientation
            # angular_velocity = imu_data.angular_velocity
            # linear_acceleration = imu_data.linear_acceleration
            # self.orientation_y = imu_data.orientation.y


            # Orientation (quaternion)
            # qx = imu_data.orientation.x
            # qy = imu_data.orientation.y
            # qz = imu_data.orientation.z
            # qw = imu_data.orientation.w

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            # roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])

            # Linear acceleration
            # ax = imu_data.linear_acceleration.x
            # ay = imu_data.linear_acceleration.y
            # az = imu_data.linear_acceleration.z

            # Angular velocity
            # gx = imu_data.angular_velocity.x
            # gy = imu_data.angular_velocity.y
            # gz = imu_data.angular_velocity.z

            # Store full IMU data: [roll, pitch, yaw, ax, ay, az, gx, gy, gz]
            # self.imu_data = np.array([roll, pitch, yaw, ax, ay, az, gx, gy, gz])

        # Print the extracted values
            # print("orientation = ", orientation_q)
            # print("angular_velocity = ", angular_velocity)
            # print("linear_acceleration = ", linear_acceleration)

    def depth_callback(self, msg):
        # print('msg', msg)
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        if cv_depth is None:
            return

        valid_mask = (cv_depth > 0) & np.isfinite(cv_depth)
        if not np.any(valid_mask):
            return

        valid_depths = cv_depth[valid_mask]
        min_depth_val = np.percentile(valid_depths, 5)
        max_depth_val = np.percentile(valid_depths, 95)
        if max_depth_val - min_depth_val < 1e-3:
            max_depth_val = min_depth_val + 1e-3

        normed = (cv_depth - min_depth_val) / (max_depth_val - min_depth_val)
        normed = np.clip(normed, 0.0, 1.0)
        normed = np.power(normed, GAMMA_VALUE)  # gamma correction
        self.normed_depth = normed.astype(np.float32)
        # print('self.normed_depth', self.normed_depth)
        # Scaled depth for collision detection
        scaled_depth = cv_depth * DISTANCE_SCALE
        # print('scaled_depth', scaled_depth)
        h, w = cv_depth.shape
        slice_width = w // ENVIRONMENT_DIM
        dist_array = np.ones(ENVIRONMENT_DIM, dtype=np.float32) * (10.0 * DISTANCE_SCALE)

        for i in range(ENVIRONMENT_DIM):
            cstart = i * slice_width
            cend = w if (i == ENVIRONMENT_DIM - 1) else (i + 1) * slice_width
            chunk = scaled_depth[:, cstart:cend]
            m = (chunk > 0) & np.isfinite(chunk)
            if np.any(m):
                dist_array[i] = chunk[m].min()

        # Minimum real distance for collision checks
        mask_scaled = (scaled_depth > 0) & np.isfinite(scaled_depth)
        if np.any(mask_scaled):
            self.min_wall_real = np.min(scaled_depth[mask_scaled])
        else:
            self.min_wall_real = 10.0

        # Optional debug display
        debug_frame = (normed * 255).astype(np.uint8)
        cv2.putText(
            debug_frame,
            f"Closest dist (scaled): {self.min_wall_real:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        # cv2.imshow("Depth Debug (Grayscale)", debug_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     print("Q pressed: resetting environment.")
        #     self.reset()

        self.realsense_data = dist_array
        # print('self.realsense_data', self.realsense_data)

    @staticmethod
    def observe_collision(dist_array):
        # print('callback_ orinentation y', orientation_y)
        # Dist array is scaled by DISTANCE_SCALE
        scaled_collision_threshold = COLLISION_DIST * DISTANCE_SCALE
        min_dist = dist_array.min()
        if min_dist < scaled_collision_threshold:
            return True, True, min_dist
        # Extra collision check: y value < 0
        # if orientation_y < -0.5 or orientation_y > 0.5:
        #     return True, True, min_dist
        return False, False, min_dist

    def step(self, action):
        self.current_step += 1
        target = False
        done = False

        # Publish velocity
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Step the simulation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            pass

        # Collision detection
        # done, collision, min_dist_array = self.observe_collision(self.realsense_data, )
        done, collision, min_dist_array = self.observe_collision(self.velodyne_data, )
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        # Extra collision check: dark blue range
        # if np.any(self.realsense_data < BLUE_DISTANCE_THRESHOLD):
        #     print("[Env] Extremely close => reset!")
        #     self.total_time = time.time() - self.episode_start_time if self.episode_start_time else 0.0
        #     self.take_screenshot()
        #     done = True
        #     collision = True
        
        if self.last_robot_x is None:
            self.last_robot_x = self.odom_x
            self.last_robot_y = self.odom_y

        dist_moved = np.linalg.norm([self.odom_x - self.last_robot_x, self.odom_y - self.last_robot_y])
        if dist_moved < STUCK_MOVEMENT_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_robot_x = self.odom_x
        self.last_robot_y = self.odom_y

        stuck = False
        self.time_step += 1
        if self.stuck_counter >= STUCK_STEPS or self.time_step == 500:
            self.total_time = time.time() - self.episode_start_time if self.episode_start_time else 0.0
            self.take_screenshot()
            print("[Env] Robot stuck => reset")
            done = True
            stuck = True
            # for i in range(30):
            #     vel_cmd = Twist()
            #     vel_cmd.linear.x = 0
            #     vel_cmd.angular.z = -0.2
            #     self.vel_pub.publish(vel_cmd)
            #     self.publish_markers(action)

            #     # Step the simulation
            #     rospy.wait_for_service("/gazebo/unpause_physics")
            #     try:
            #         self.unpause()
            #     except rospy.ServiceException:
            #         pass
            #     time.sleep(TIME_DELTA)
            #     rospy.wait_for_service("/gazebo/pause_physics")
            #     try:
            #         self.pause()
            #     except rospy.ServiceException:
            #         pass

        # Near wall detection
        if min_dist_array < (1.0 * DISTANCE_SCALE):
            self.near_wall_counter += 1
        else:
            self.near_wall_counter = 0
        if self.near_wall_counter >= NEAR_WALL_STEPS:
            self.total_time = time.time() - self.episode_start_time if self.episode_start_time else 0.0
            self.take_screenshot()
            print("[Env] near wall => reset")
            done = True

        # Check goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        if distance < GOAL_REACHED_DIST:
            self.total_time = time.time() - self.episode_start_time if self.episode_start_time else 0.0
            self.take_screenshot()
            target = True
            done = True

        if self.prev_distance is None:
            self.prev_distance = distance
        if self.prev_wall_real is None:
            self.prev_wall_real = self.min_wall_real

        # self.step_count += 1
        # if done:
        #     self.step_count = 0
        # elif self.step_count > MAX_STEPS and done is False:
        #     done = True
        #     stuck = True
            # target = False
        if self.current_step >= self.max_steps:
            self.total_time = time.time() - self.episode_start_time if self.episode_start_time else 0.0
            self.take_screenshot()
            print("[Env] Max steps reached => ending episode")
            done = True
            stuck = True
        # Compute reward
        # final_r = 0
        angle = self.compute_angle_to_goal()
        # final_r = self.get_reward(
        #     target, collision, stuck,
        #     action,
        #     distance, self.prev_distance,
        #     angle,
        #     self.min_wall_real, self.prev_wall_real
        # )
        # else:
        #     final_r = self.get_reward_phase_2(
        #         target, collision, stuck,
        #         action,
        #         distance, self.prev_distance,
        #         angle,
        #         self.min_wall_real, self.prev_wall_real
        #     )
        # print(f"[DEBUG] done={done}, collision={collision}, stuck={stuck}, reward={final_r:.2f}")

        self.prev_distance = distance
        self.prev_wall_real = self.min_wall_real

        # Build observation
        self.prev_action = self.last_action
        self.last_action = np.array([action[0], action[1]], dtype=np.float32)

        dist2goal = distance
        angle2goal = angle
        min_laser = min_dist_array
        reward = self.get_reward(target, collision, action, min_laser)
        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            dist2goal,
            angle2goal,
            # min_laser,
        ], dtype=np.float32)

        # single-channel (64x64) depth
        if self.normed_depth is not None:
            resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
            one_channel = resized[None, ...]  # shape (1,64,64)
        else:
            one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        
        return (laser_state, scalars), reward, done, target

    def reset(self):
        # self.clear_path()

        self.current_step = 0
        self.episode_num += 1
        self.stuck_counter = 0
        self.last_robot_x = None
        self.last_robot_y = None
        self.near_wall_counter = 0

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_distance = None
        self.prev_wall_real = None
        self.min_wall_real = 10.0
        self.time_step = 0
        # Reset cumulative rotation and last_yaw as well
        self.cum_rotation = 0.0
        self.last_yaw = None
        self.episode_start_time = time.time() 
        self.episode_sim_time = 0.0   
        # rospy.wait_for_service("/gazebo/reset_world")
        # try:
        #     self.reset_proxy()
        # except rospy.ServiceException:
        #     pass

        # #vRandomize position
        angle = np.random.uniform(-math.pi, math.pi)
        quat = Quaternion.from_euler(0.0, 0.0, angle)
        x_ok = False
        x = 0
        y = 0
        rooms = [
            (-3, 2, 4, 8),        # Room 1
            (-3, 1, 1.5, 3),      # Room 2
            (-2.5, 2, -3.5, 0.5), # Room 3
            (-2.5, 0.5, -6, -5),  # Room 4
            (2, 5, -6, -5),       # Room 5
            (3, 7, 6.5, 7.5),     # Room 6
        ]
        
        x_ok = False
        attempts = 0
        
        while not x_ok and attempts < 500:
            # Pick random room
            room = random.choice(rooms)
            
            # Sample position in that room
            x = np.random.uniform(room[0], room[1])
            y = np.random.uniform(room[2], room[3])
            
            x_ok = self.check_pos(x, y)
            attempts += 1



        st = self.set_self_state
        st.pose.position.x = x
        st.pose.position.y = y
        st.pose.position.z = 0
        st.pose.orientation.x = quat.x
        st.pose.orientation.y = quat.y
        st.pose.orientation.z = quat.z
        st.pose.orientation.w = quat.w
        self.set_state.publish(st)

        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = angle

        # self.random_box()
        self.change_goal()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            pass

        rospy.sleep(0.2)

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        self.prev_distance = distance

        angle_to_goal = self.compute_angle_to_goal()
        # min_laser = self.min_wall_real
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            distance,
            angle_to_goal,
            # min_laser,
        ], dtype=np.float32)

        # 1-channel depth
        # if self.normed_depth is not None:
        #     resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
        #     one_channel = resized[None, ...]
        # else:
        #     one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        return (laser_state, scalars)

    def check_pos(self, x, y):
        goal_ok = True

        if -2 > x > 0.5 and 4.7 > y > 7.2:
            goal_ok = False

        if -1 > x > 0 and -2.5 > y > -1:
            goal_ok = False

        if 3.7 > x > 6 and 7 > y > 8:
            goal_ok = False

        if -3.5 > x > -2.8 and 5.5 > y > 7:
            goal_ok = False

        if -3 > x > -2.5 and -4 > y > -3.5:
            goal_ok = False

        if -0.6 > x > 0.6 and 2.7 > y > 3:
            goal_ok = False

        # if 4.1 > x > 2.6 and 0.8 > y > -3.3:
        #     goal_ok = False

        # if 6.3 > x > 3.9 and -3.4 > y > -4.3:
        #     goal_ok = False

        # if 4.3 > x > 1.5 and 3.9 > y > 1.7:
        #     goal_ok = False

        # if -3.2 > x > -7.4 and 0.7 > y > -1.7:
        #     goal_ok = False

        # if x > 4.7 or x < -4.7 or y > 4.7 or y < -4.5:
        #     goal_ok = False

         # Check collision with boxes
        for bx, by in getattr(self, "box_positions", []):
            if np.linalg.norm([x - bx, y - by]) < 1:
                goal_ok = False
                break

        return goal_ok

    def change_goal(self):
        x = 0
        y = 0
        rooms = [
            (-3, 2, 4, 8),        # Room 1
            (-3, 1, 1.5, 3),      # Room 2
            (-2.5, 2, -3.5, 0.5), # Room 3
            (-2.5, 0.5, -6, -5),  # Room 4
            (2, 5, -6, -5),       # Room 5
            (3, 7, 6.5, 7.5),     # Room 6
        ]
        
        x_ok = False
        attempts = 0
        
        while not x_ok and attempts < 500:
            # Pick random room
            room = random.choice(rooms)
            
            # Sample position in that room
            x = np.random.uniform(room[0], room[1])
            y = np.random.uniform(room[2], room[3])
            
            x_ok = self.check_pos(x, y)
            attempts += 1
        self.goal_x = x
        self.goal_y = y



    def publish_markers(self, action):
        # Visual markers for debugging
        markerArray = MarkerArray()
        # Goal marker
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher_goal.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = Marker.CUBE
        marker2.action = Marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0
        marker2.color.b = 0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0
        markerArray2.markers.append(marker2)
        self.publisher_lin.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = Marker.CUBE
        marker3.action = Marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0
        marker3.color.b = 0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0
        markerArray3.markers.append(marker3)
        self.publisher_ang.publish(markerArray3)

    def compute_angle_to_goal(self):
        dx = self.goal_x - self.odom_x
        dy = self.goal_y - self.odom_y
        desired_angle = math.atan2(dy, dx)
        angle_to_goal = desired_angle - self.odom_yaw
        # Wrap to [-pi, pi]
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        return angle_to_goal


    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
        
    # def get_reward(
    #     self,
    #     target,
    #     collision,
    #     stuck,
    #     action,
    #     distance,
    #     prev_distance,
    #     angle,
    #     current_wall_real,
    #     prev_wall_real
    # ):
    #     """
    #     An improved reward function that:
    #       - Rewards progress (delta in distance to goal)
    #       - Gives angle bonus for facing the goal
    #       - Penalizes collisions/stuck
    #       - Adds spin penalty if spinning in place
    #       - Adds a penalty for excessive cumulative rotation
    #       - Adds a step penalty to encourage faster completion
    #     """
    #     SLIDE_TOLERANCE = 0.1  # Acceptable deviation from y=0
    #     SLIDE_PENALTY_COEFF = 5.0

    #     gp = 0.0  # goal progress
    #     # Large reward for reaching goal / large penalty for collision
    #     if target:
    #         return 100.0
    #     if collision or stuck:
    #         return -100.0

    #     # 1) Base motion term: reward for forward speed, penalty for high angular speed.
    #     base = (action[0] / 2) - (abs(action[1]) / 2) 

    #     # 2) Progress reward: scaled difference in distance (positive if closer, negative if farther).
    #     progress_coeff = 1.0 # неге 3.0 агай? 
    #     progress = progress_coeff * (prev_distance - distance)

    #     # 3) Angle bonus: encourage facing the goal.
    #     alpha = 1.0
    #     angle_bonus = alpha * (1.0 - (abs(angle) / math.pi))

    #     # 4) Wall penalty: penalize both proximity AND approaching
    #     wall_penalty = 0.0

    #     # A) Static proximity penalty (like r3)
    #     SAFE_DISTANCE = 1.0
    #     if current_wall_real < SAFE_DISTANCE:
    #         proximity_penalty = -(SAFE_DISTANCE - current_wall_real) / 2.0
    #     else:
    #         proximity_penalty = 0.0

    #     # B) Dynamic approach penalty (like yours) 
    #     dist_decreased = prev_wall_real - current_wall_real
    #     if dist_decreased > 0:  # Getting closer
    #         approach_penalty = -dist_decreased * 2.0  # Scale factor
    #     else:
    #         approach_penalty = 0.0

    #     # Combine both
    #     wall_penalty = proximity_penalty + approach_penalty

    #     # 5) Spin penalty: discourage in-place rotation.
    #     spin_penalty = 0.0
    #     if abs(action[0]) < 0.02 and abs(action[1]) > 0.2:
    #         spin_penalty = -0.02

    #     # 6) Small time-step penalty.
    #     step_penalty = TIME_STEP_PENALTY

    #     # 7) Cumulative rotation penalty: penalize excessive turning.
    #     rotation_penalty = 0.0
    #     if self.cum_rotation > 1.0:
    #         rotation_penalty = -0.1 * self.cum_rotation
    #         # Optionally reset cumulative rotation after applying the penalty
    #         self.cum_rotation = 0.0


    #     # backward penalty
    #     backward_penalty = 0.0
    #     if action[0] < 0:
    #         backward_penalty = -1
    #     # 8) Lateral sliding penalty (penalize deviation from y = 0)
    #     # slide_penalty = 0.0
    #     # if abs(self.imu_data[1]) > SLIDE_TOLERANCE:
    #     #     slide_penalty = -SLIDE_PENALTY_COEFF * abs(self.imu_data[1])
    #     # In get_reward_phase_1
    #     action_smoothness_penalty = -0.1 * np.linalg.norm(action - self.last_action)


    #     total = (
    #         base
    #         # + progress
    #         # + angle_bonus
    #         # + spin_penalty
    #         # + step_penalty
    #         # + rotation_penalty
    #         # + action_smoothness_penalty
    #         + wall_penalty 
    #         # + 2 * backward_penalty
    #     )

   

    #     return total


    # def init_path_marker(self):
    #     """Initialize path tracking and overhead camera"""
    #     self._screen_count = 1
    #     self.overhead_camera_image = None
    #     self.screenshot_dir = '/tmp/exploration_data/from_top_screen/'
    #     os.makedirs(self.screenshot_dir, exist_ok=True)

    #     self.camera_rotation = 3

    #     self.map_x_min = -5.5
    #     self.map_x_max = 5.5
    #     self.map_y_min = -5.5
    #     self.map_y_max = 5.5

    #     self.overhead_cam_sub = rospy.Subscriber(
    #         "/overhead_camera/image_raw", 
    #         Image, 
    #         self.overhead_camera_callback, 
    #         queue_size=1
    #     )
        
    #     # Path tracking for current trial
    #     self.path_points_xy = []

    #     self.all_trial_paths = []  # List of (path_points, color, result, trial_num)
    #     self.current_path_color = self.generate_random_color()
        
    #     # RViz marker
    #     self.marker_pub = rospy.Publisher('/robot_path', Marker, queue_size=10)
    #     self.marker = Marker()
    #     self.marker.header.frame_id = "odom"
    #     self.marker.ns = "robot_path"
    #     self.marker.id = 0
    #     self.marker.type = Marker.LINE_STRIP
    #     self.marker.action = Marker.ADD
    #     self.marker.scale.x = 0.05
    #     self.marker.color.r = 1.0
    #     self.marker.color.g = 0.0
    #     self.marker.color.b = 0.0
    #     self.marker.color.a = 1.0
    #     self.marker.pose.orientation.w = 1.0
    #     self.marker.lifetime = rospy.Duration(0)
    #     self.path_points = []

    # def generate_random_color(self):
    #     """Generate a random bright color (BGR format for OpenCV)."""
        
    #     hue = random.randint(0, 180)
    #     saturation = random.randint(200, 255)
    #     value = random.randint(200, 255)
        
    #     # Create a 1x1 HSV image and convert to BGR
    #     hsv_color = np.uint8([[[hue, saturation, value]]])
    #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        
    #     # Return as tuple (B, G, R)
    #     return tuple(int(c) for c in bgr_color[0][0])

    # def overhead_camera_callback(self, msg):
    #     """Callback for overhead bird's eye view camera"""
    #     self.overhead_camera_image = msg

    # def take_screenshot(self, episode_result="goal"):
    #     """Take screenshot with ALL trial paths drawn on it."""
    #     os.makedirs(self.screenshot_dir, exist_ok=True)
        
    #     if self.overhead_camera_image is None:
    #         rospy.logwarn("No image from overhead camera yet.")
    #         return None
        
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(self.overhead_camera_image, "bgr8")
    #         img_height, img_width = cv_image.shape[:2]
            
    #         # Helper function for coordinate conversion
    #         def to_pixel(wx, wy):
    #             return self.world_to_image(wx, wy, img_width, img_height)
            
    #         if len(self.path_points_xy) >= 2:
    #             self.all_trial_paths.append({
    #                 'points': list(self.path_points_xy),  # Copy the path
    #                 'color': self.current_path_color,
    #                 'result': episode_result,
    #                 'trial_num': self.episode_num
    #             })
            
    #         for trial in self.all_trial_paths:
    #             path_points = trial['points']
    #             path_color = trial['color']
                
    #             if len(path_points) >= 2:
    #                 pixel_points = [to_pixel(wx, wy) for (wx, wy) in path_points]
                    
    #                 # Draw path
    #                 for i in range(1, len(pixel_points)):
    #                     cv2.line(cv_image, pixel_points[i-1], pixel_points[i], 
    #                             path_color, 2)
                    
    #                 # Draw start point (small green)
    #                 cv2.circle(cv_image, pixel_points[0], 6, (0, 255, 0), -1)
                    
    #                 # Draw end point (small, path color)
    #                 end_color = tuple(max(0, c - 50) for c in path_color)
    #                 cv2.circle(cv_image, pixel_points[-1], 6, end_color, -1)
            
    #         goal_px, goal_py = to_pixel(self.goal_x, self.goal_y)
    #         cv2.circle(cv_image, (goal_px, goal_py), 15, (0, 255, 255), -1)
    #         cv2.circle(cv_image, (goal_px, goal_py), 17, (255, 255, 255), 2)
    #         cv2.drawMarker(cv_image, (goal_px, goal_py), (0, 0, 0), 
    #                     cv2.MARKER_CROSS, 20, 2)
            

    #         robot_px, robot_py = to_pixel(self.odom_x, self.odom_y)
    #         cv2.circle(cv_image, (robot_px, robot_py), 8, (255, 0, 255), -1)
            

    #         def put_text_with_bg(img, text, pos, font_scale=0.6, color=(255,255,255)):
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             thickness = 2
    #             (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    #             x, y = pos
    #             cv2.rectangle(img, (x-2, y-h-5), (x+w+2, y+5), (0,0,0), -1)
    #             cv2.putText(img, text, pos, font, font_scale, color, thickness)
            
    #         # Count successes and failures
    #         successes = sum(1 for t in self.all_trial_paths if t['result'] == 'goal')
    #         total_trials = len(self.all_trial_paths)
            
    #         put_text_with_bg(cv_image, f"Trials: {total_trials} | Success: {successes}/{total_trials}", (10, 25))
    #         put_text_with_bg(cv_image, f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})", (10, 50))
            
    #         # ============================================
    #         # LEGEND WITH TRIAL COLORS
    #         # ============================================
    #         legend_y = img_height - 20
    #         legend_x = 10
            
    #         # Draw legend for each trial (max 10 shown)
    #         trials_to_show = self.all_trial_paths[-10:]  # Last 10 trials
    #         for i, trial in enumerate(trials_to_show):
    #             ly = legend_y - (len(trials_to_show) - i) * 15
    #             result_symbol = "✓" if trial['result'] == 'goal' else "✗"
    #             cv2.circle(cv_image, (legend_x, ly), 5, trial['color'], -1)
                
            
    #         # Goal marker in legend
    #         cv2.circle(cv_image, (img_width - 60, legend_y), 6, (0, 255, 255), -1)
    #         cv2.putText(cv_image, "Goal", (img_width - 45, legend_y + 4), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
      
    #         combined_filename = os.path.join(self.screenshot_dir, 
    #                             f'combined_goal_{self.goal_x:.1f}_{self.goal_y:.1f}.png')
    #         cv2.imwrite(combined_filename, cv_image)
            
    #         # Also save individual trial screenshot
    #         individual_filename = os.path.join(self.screenshot_dir, 
    #                             f'ep{self.episode_num}_{episode_result}.png')
    #         cv2.imwrite(individual_filename, cv_image)
            
    #         rospy.loginfo(f"✅ Combined screenshot saved: {combined_filename} ({total_trials} trials)")
    #         return combined_filename
            
    #     except CvBridgeError as e:
    #         rospy.logerr(f"CV Bridge error: {e}")
    #         return None
    
    # def clear_all_trials(self):
    #     """Clear all stored trial paths - call when starting a new goal"""
    #     self.all_trial_paths = []
            
    # def clear_path(self):
    #     """Clear the recorded path for new episode and store for combined image"""
        
    #     self.path_points_xy = []
    #     self.path_points = []
        
    #     # Generate new random color for next trial
    #     self.current_path_color = self.generate_random_color()
        
    #     self.marker.points = []
    #     self.marker.action = Marker.DELETEALL
    #     self.marker_pub.publish(self.marker)
    #     self.marker.action = Marker.ADD

    # def world_to_image(self, world_x, world_y, img_width, img_height):
        """
        Convert world coordinates to image pixel coordinates.
        """
        camera_x_min = -5.5
        camera_x_max = 5.5
        camera_y_min = -5.5
        camera_y_max = 5.5
        
        # Normalize world coordinates to 0-1 within camera visible area
        norm_x = (world_x - camera_x_min) / (camera_x_max - camera_x_min)
        norm_y = (world_y - camera_y_min) / (camera_y_max - camera_y_min)
        
        # Clamp to 0-1
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Apply rotation 3: world X = image up, world Y = image left
        rotated_norm_x = 1 - norm_y
        rotated_norm_y = 1 - norm_x
        
        # Map directly to image pixels (camera sees entire image)
        pixel_x = int(rotated_norm_x * img_width)
        pixel_y = int(rotated_norm_y * img_height)
        
        # Clamp to image bounds
        pixel_x = max(0, min(img_width - 1, pixel_x))
        pixel_y = max(0, min(img_height - 1, pixel_y))
        
        return pixel_x, pixel_y