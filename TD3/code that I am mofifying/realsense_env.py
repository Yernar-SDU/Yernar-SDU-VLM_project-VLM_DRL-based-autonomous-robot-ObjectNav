#!/usr/bin/env python3
import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

###############################################
#          ALL ADJUSTABLE PARAMETERS
###############################################
LAUNCHFILE = "multi_robot_scenario.launch"

# WIDER beam coverage
ENVIRONMENT_DIM = 36
TIME_DELTA = 0.1

GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.35

# For "stuck" detection:
STUCK_STEPS = 30
STUCK_MOVEMENT_THRESHOLD = 0.02

NEAR_WALL_STEPS = 20
DISTANCE_SCALE = 0.01
BLUE_DISTANCE_THRESHOLD = 0.15

# Slightly bigger penalty each step
TIME_STEP_PENALTY = -0.05

# The depth image shape (height=64, width=128)
IMG_HEIGHT = 64
IMG_WIDTH = 128

# Gamma correction
GAMMA_VALUE = 0.8

###############################################

class GazeboEnv:
    """
    Now returns a SINGLE depth-based channel of shape (1,64,128)
    plus a 7D array of scalars:
      [prev_lin, prev_ang, last_lin, last_ang, dist2goal, angle2goal, min_laser]
    Also uses a double progress reward and bigger angle bonus.
    """

    def __init__(self):
        self.environment_dim = ENVIRONMENT_DIM
        self.odom_x = 0
        self.odom_y = 0
        self.odom_yaw = 0.0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        # Distances from RealSense depth array
        self.realsense_data = np.ones(self.environment_dim, dtype=np.float32) * (10.0 * DISTANCE_SCALE)

        # We'll store the single normalized depth image in self.normed_depth
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
        # For "wall" reward from the depth camera:
        self.prev_wall_real = None

        # For set_model_state
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"

        # Launch roscore
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        rospy.init_node("gym", anonymous=True)
        if LAUNCHFILE.startswith("/"):
            fullpath = LAUNCHFILE
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", LAUNCHFILE)
        if not path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched with", LAUNCHFILE)

        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.publisher_goal = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher_lin = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher_ang = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber("/realsense_camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)

        rospy.sleep(2.0)
        print("Environment ready!")

    def odom_callback(self, od_data):
        self.last_odom = od_data
        self.odom_x = od_data.pose.pose.position.x
        self.odom_y = od_data.pose.pose.position.y
        orientation_q = od_data.pose.pose.orientation
        quat = Quaternion(orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z)
        _, _, yaw = quat.to_euler()
        self.odom_yaw = yaw

    def depth_callback(self, msg):
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
        normed = np.power(normed, GAMMA_VALUE)
        self.normed_depth = normed.astype(np.float32)

        # Build the beam array
        scaled_depth = cv_depth * DISTANCE_SCALE
        h, w = cv_depth.shape
        slice_width = w // self.environment_dim
        dist_array = np.ones(self.environment_dim, dtype=np.float32) * (10.0 * DISTANCE_SCALE)
        for i in range(self.environment_dim):
            cstart = i * slice_width
            cend = w if (i == self.environment_dim - 1) else (i + 1) * slice_width
            chunk = scaled_depth[:, cstart:cend]
            m = (chunk > 0) & np.isfinite(chunk)
            if np.any(m):
                dist_array[i] = chunk[m].min()
        self.realsense_data = dist_array

        # min distance
        mask_scaled = (scaled_depth > 0) & np.isfinite(scaled_depth)
        if np.any(mask_scaled):
            min_scaled = np.min(scaled_depth[mask_scaled])
        else:
            min_scaled = 10.0

        self.min_wall_real = min_scaled

        # debug view
        debug_frame = (normed * 255).astype(np.uint8)
        cv2.putText(
            debug_frame,
            f"Closest dist (scaled): {min_scaled:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.imshow("Depth Debug (Grayscale)", debug_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("q pressed: resetting environment")
            self.reset()

    @staticmethod
    def observe_collision(dist_array):
        scaled_collision = COLLISION_DIST * DISTANCE_SCALE
        min_dist = dist_array.min()
        if min_dist < scaled_collision:
            return True, True, min_dist
        return False, False, min_dist

    def step(self, action):
        target = False

        # apply the action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # step sim
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

        # collision detection
        done, collision, min_dist_array = self.observe_collision(self.realsense_data)

        if np.any(self.realsense_data < BLUE_DISTANCE_THRESHOLD):
            print("[Env] Dangerously close => reset!")
            done = True
            collision = True

        # stuck detection
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
        if self.stuck_counter >= STUCK_STEPS:
            print("[Env] Robot stuck => reset")
            done = True
            stuck = True

        # near wall steps
        if min_dist_array < (1.0 * DISTANCE_SCALE):
            self.near_wall_counter += 1
        else:
            self.near_wall_counter = 0
        if self.near_wall_counter >= NEAR_WALL_STEPS:
            print("[Env] near wall => reset")
            done = True

        # check goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        if self.prev_distance is None:
            self.prev_distance = distance
        if not hasattr(self, 'min_wall_real'):
            self.min_wall_real = 10.0
        if self.prev_wall_real is None:
            self.prev_wall_real = self.min_wall_real

        # compute reward
        angle = self.compute_angle_to_goal()
        (base_r, prog_r, wall1, angle_b, wall2, total_r) = self.get_reward(
            target, collision, stuck,
            action,
            distance, self.prev_distance,
            angle,
            self.min_wall_real, self.prev_wall_real
        )

        # add time step penalty
        total_r += TIME_STEP_PENALTY

        # debug
        print(f"[REWARD DETAILS] base={base_r:.2f}, progress={prog_r:.2f}, wall1={wall1:.2f}, "
              f"angle={angle_b:.2f}, wall2={wall2:.2f}, final={total_r:.2f}")
        print(f"[DEBUG] done={done}, collision={collision}, stuck={stuck}, reward={total_r:.2f}")

        self.prev_distance = distance
        self.prev_wall_real = self.min_wall_real

        # build next observation
        self.prev_action = self.last_action
        self.last_action = np.array([action[0], action[1]], dtype=np.float32)

        dist2goal = distance
        angle_to_goal = angle
        min_laser = min_dist_array

        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            dist2goal,
            angle_to_goal,
            min_laser
        ], dtype=np.float32)

        if self.normed_depth is not None:
            resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
            one_channel = resized[None, ...]  # shape (1,64,128)
        else:
            one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        return (one_channel, scalars), total_r, done, target

    def reset(self):
        self.stuck_counter = 0
        self.last_robot_x = None
        self.last_robot_y = None
        self.near_wall_counter = 0

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_distance = None
        self.prev_wall_real = None

        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            pass

        angle = np.random.uniform(-math.pi, math.pi)
        quat = Quaternion.from_euler(0, 0, angle)
        x_ok = False
        x = 0
        y = 0
        while not x_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            x_ok = self.check_pos(x, y)

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

        self.change_goal()
        self.random_box()
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

        if not hasattr(self, 'normed_depth'):
            self.normed_depth = None

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        if not hasattr(self, 'min_wall_real'):
            self.min_wall_real = 10.0
        self.prev_distance = distance
        self.prev_wall_real = self.min_wall_real

        angle_to_goal = self.compute_angle_to_goal()
        min_laser = self.realsense_data.min()

        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            distance,
            angle_to_goal,
            min_laser
        ], dtype=np.float32)

        if self.normed_depth is not None:
            resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
            one_channel = resized[None, ...]
        else:
            one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        return (one_channel, scalars)

    def check_pos(self, x, y):
        # same obstacle checks
        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            return False
        if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
            return False
        return True

    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        ok = False
        while not ok:
            gx = self.odom_x + random.uniform(self.upper, self.lower)
            gy = self.odom_y + random.uniform(self.upper, self.lower)
            ok = self.check_pos(gx, gy)
            if ok:
                self.goal_x = gx
                self.goal_y = gy

    def random_box(self):
        for i in range(4):
            name = f"cardboard_box_{i}"
            box_ok = False
            while not box_ok:
                xx = np.random.uniform(-6, 6)
                yy = np.random.uniform(-6, 6)
                box_ok = self.check_pos(xx, yy)
                dist_robot = np.linalg.norm([xx - self.odom_x, yy - self.odom_y])
                dist_goal = np.linalg.norm([xx - self.goal_x, yy - self.goal_y])
                if dist_robot < 1.5 or dist_goal < 1.5:
                    box_ok = False
            st = ModelState()
            st.model_name = name
            st.pose.position.x = xx
            st.pose.position.y = yy
            st.pose.position.z = 0
            st.pose.orientation.x = 0
            st.pose.orientation.y = 0
            st.pose.orientation.z = 0
            st.pose.orientation.w = 1
            self.set_state.publish(st)

    def publish_markers(self, action):
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

        # Linear velocity marker
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

        # Angular velocity marker
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
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        return angle_to_goal

    @staticmethod
    def get_reward(
        target, collision, stuck,
        action,
        distance, prev_distance,
        angle,
        current_wall_real, prev_wall_real
    ):
        """
        Double the progress. Also scale up angle bonus.
        """
        if target:
            return (0, 0, 0, 0, 0, 1000.0)
        elif collision or stuck:
            return (0, 0, 0, 0, 0, -1500.0)
        else:
            base = (action[0] / 2) - (abs(action[1]) / 2)
            
            # Double progress
            progress = 2.0 * (prev_distance - distance)

            # wall1 if <2.5
            wall1 = 0.0
            if current_wall_real < 2.5:
                wall1 = -(2.5 - current_wall_real)

            # angle bonus scaled up
            alpha = 2.0
            angle_bonus = alpha * (1.0 - (abs(angle) / math.pi))

            dist_decreased = (prev_wall_real - current_wall_real)
            wall2 = 0.0
            if dist_decreased > 0:
                wall2 = -1.0 * dist_decreased

            total = 3*base + progress + wall1 + angle_bonus + wall2
            return (base, progress, wall1, angle_bonus, wall2, total)


if __name__ == "__main__":
    env = GazeboEnv()
    obs = env.reset()  # obs = (image, scalars)
    while not rospy.is_shutdown():
        # random action
        action = [random.uniform(0.0, 0.6), random.uniform(-1.0, 1.0)]
        obs, reward, done, _ = env.step(action)
        if done:
            env.reset()
        time.sleep(0.1)
