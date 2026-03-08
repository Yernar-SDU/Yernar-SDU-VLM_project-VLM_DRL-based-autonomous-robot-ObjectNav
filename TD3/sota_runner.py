
import time
import math
import os
import subprocess
import threading
import numpy as np
import pandas as pd
import rospy
import actionlib
import tf
import json
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from squaternion import Quaternion

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
LAUNCHFILE = "multi_robot_scenario.launch"
EXCEL_FILE = "navigation_trials_sota_1.xlsx"
OBJECTS_JSONL = "/tmp/exploration_data/detected_objects.jsonl"
SUCCESS_RADIUS = 0.9
GOAL_TIMEOUT = 60.0
TRIALS_PER_GOAL = 10
COLLISION_DISTANCE = 0.35
ROBOT_MODEL_NAME = "r1"  # ← CORRECT NAME!

NEEDS_MOVE_BASE = {"dwa", "teb", "navfn"}


# ═══════════════════════════════════════════════════════════════════════════════
# ODOMETRY BROADCASTER (Fixes the broken Gazebo odom)
# ═══════════════════════════════════════════════════════════════════════════════
class OdometryBroadcaster:
    def __init__(self, robot_name="r1"):
        self.robot_name = robot_name
        self.tf_broadcaster = tf.TransformBroadcaster()
        # Ensure we publish to the topic move_base will eventually listen to
        self.odom_pub = rospy.Publisher("/r1/odom_custom", Odometry, queue_size=10)
        
        self.x = self.y = self.z = 0.0
        self.quat = (0, 0, 0, 1)
        self.last_stamp = rospy.Time(0)
        self.running = True
        
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback)
        rospy.loginfo(f"✅ Broadcaster started for {robot_name}")

    def _callback(self, msg):
        if not self.running: return
        try:
            idx = msg.name.index(self.robot_name)
            p = msg.pose[idx]
            t = msg.twist[idx]
            
            self.x, self.y, self.z = p.position.x, p.position.y, p.position.z
            self.quat = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            
            # Synchronize with Sim Time
            curr_time = rospy.Time.now()
            if curr_time <= self.last_stamp:
                curr_time = self.last_stamp + rospy.Duration(0.001)
            self.last_stamp = curr_time

            # FIX 1: The 'Glue' that holds the robot together in the world
            self.tf_broadcaster.sendTransform(
                (self.x, self.y, self.z), self.quat, curr_time, "base_link", "odom"
            )

            # FIX 2: Create the Odom message move_base is starving for
            om = Odometry()
            om.header.stamp = curr_time
            om.header.frame_id = "odom"
            om.child_frame_id = "base_link"
            om.pose.pose = p
            om.twist.twist = t
            self.odom_pub.publish(om)
            
        except ValueError: pass

    def _publish_tf_and_odom(self):
        current_time = rospy.Time.now()
        if current_time <= self.last_stamp:
            current_time = self.last_stamp + rospy.Duration(0.001)
        self.last_stamp = current_time
        
        # FORCE the link that keeps the robot together
        # Child must be "base_link" (no prefix)
        # Parent must be "odom" (no prefix)
        self.tf_broadcaster.sendTransform(
            (self.x, self.y, self.z),
            self.quat,
            current_time,
            "base_link", 
            "odom"
        )
        
        # Publish message for move_base
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        # 2. Publish Odometry Message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation.x = self.quat[0]
        odom.pose.pose.orientation.y = self.quat[1]
        odom.pose.pose.orientation.z = self.quat[2]
        odom.pose.pose.orientation.w = self.quat[3]
        
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.angular.z = self.wz
        
        self.odom_pub.publish(odom)
    
    def get_pose(self):
        """Used by GazeboInterface and SOTARunner"""
        return self.x, self.y, self.quat

    def get_yaw(self):
        """Helper to get yaw from quaternion"""
        q = Quaternion(self.quat[3], self.quat[0], self.quat[1], self.quat[2])
        _, _, yaw = q.to_euler()
        return yaw

    def stop(self):
        """Called during SOTARunner shutdown"""
        self.running = False
        self.sub.unregister()
# ═══════════════════════════════════════════════════════════════════════════════
# GAZEBO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
class GazeboInterface:
    """Clean interface to Gazebo simulation"""
    
    def __init__(self):
        self.robot_name = ROBOT_MODEL_NAME
        
        # Robot state (will be updated by OdometryBroadcaster)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        
        # Laser data
        self.laser_ranges = []
        self.laser_angle_min = 0.0
        self.laser_angle_increment = 0.0
        self.is_collision = False
        # Path tracking
        self.path_points = []
        self.path_length = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        
        # Kill existing processes
        self._kill_ros_processes()
        time.sleep(1)
        
        # Launch ROS and Gazebo
        self._launch_system()
        
        # Setup ROS interface
        self._setup_ros()
        
        # Start our odometry broadcaster (FIXES THE BROKEN GAZEBO ODOM)
        self.odom_broadcaster = OdometryBroadcaster(self.robot_name)
        
        # Wait for everything to stabilize
        rospy.sleep(2.0)
        
        print("✅ GazeboInterface ready!")
    def check_collision(self):
        """Check if robot has collided with obstacle"""
        return self.is_collision
    
    def _kill_ros_processes(self):
        """Kill existing ROS/Gazebo processes"""
        for proc in ["roscore", "rosmaster", "gzserver", "gzclient", "move_base", "rviz"]:
            subprocess.run(["killall", "-9", proc], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
    
    def _launch_system(self):
        """Launch roscore and Gazebo"""
        # Roscore
        subprocess.Popen(["roscore"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ Roscore launched")
        time.sleep(2)
        
        # Initialize ROS node
        rospy.init_node("sota_runner", anonymous=True)
        
        # Gazebo
        script_dir = os.path.dirname(os.path.realpath(__file__))
        launch_path = os.path.join(script_dir, "assets", LAUNCHFILE)
        
        if not os.path.exists(launch_path):
            raise IOError(f"Launch file not found: {launch_path}")
        
        subprocess.Popen(["roslaunch", launch_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        print("✅ Gazebo launched")
        time.sleep(8)
    
    def _setup_ros(self):
        """Setup ROS publishers, subscribers, services"""
        # Publishers
        self.cmd_vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        
        # Services
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=10)
        rospy.wait_for_service("/gazebo/pause_physics", timeout=10)
        self.unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=5)
            self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        except:
            self.set_model_state_srv = None
        
        # Subscribers
        rospy.Subscriber("/r1/front_laser/scan", LaserScan, self._laser_callback, queue_size=1)
        
        # Unpause to start
        self.unpause()
        rospy.sleep(1.0)
    
    def _laser_callback(self, msg):
        """Handle laser scan with directional collision detection"""
        self.laser_ranges = list(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_increment = msg.angle_increment
        
        n = len(self.laser_ranges)
        if n == 0:
            return
        
        def get_sector_min(start_deg, end_deg):
            """Get minimum distance in a sector (degrees from front)"""
            # Convert degrees to indices
            # Assuming laser covers 360° with index 0 = front
            start_idx = int((start_deg + 180) / 360 * n) % n
            end_idx = int((end_deg + 180) / 360 * n) % n
            
            if start_idx < end_idx:
                indices = range(start_idx, end_idx)
            else:
                indices = list(range(start_idx, n)) + list(range(0, end_idx))
            
            vals = [self.laser_ranges[i] for i in indices 
                    if math.isfinite(self.laser_ranges[i]) and self.laser_ranges[i] > 0.05]
            return min(vals) if vals else float('inf')
        
        # Check different sectors
        self.front_min_distance = get_sector_min(-45, 45)    # Front ±45°
        self.back_min_distance = get_sector_min(135, -135)   # Back ±45°
        self.left_min_distance = get_sector_min(45, 135)     # Left
        self.right_min_distance = get_sector_min(-135, -45)  # Right
        
        # Overall minimum
        valid_ranges = [r for r in self.laser_ranges if math.isfinite(r) and r > 0.05]
        self.min_laser_distance = min(valid_ranges) if valid_ranges else float('inf')
        
        # Collision in ANY direction
        if self.min_laser_distance < COLLISION_DISTANCE:
            self.is_collision = True
            
            # Log which direction
            if self.front_min_distance < COLLISION_DISTANCE:
                rospy.logwarn(f"🚨 FRONT COLLISION! {self.front_min_distance:.2f}m")
            if self.back_min_distance < COLLISION_DISTANCE:
                rospy.logwarn(f"🚨 BACK COLLISION! {self.back_min_distance:.2f}m")
            if self.left_min_distance < COLLISION_DISTANCE:
                rospy.logwarn(f"🚨 LEFT COLLISION! {self.left_min_distance:.2f}m")
            if self.right_min_distance < COLLISION_DISTANCE:
                rospy.logwarn(f"🚨 RIGHT COLLISION! {self.right_min_distance:.2f}m")
    def _update_odom_from_broadcaster(self):
        """Update internal odom state from broadcaster"""
        x, y, quat = self.odom_broadcaster.get_pose()
        self.odom_x = x
        self.odom_y = y
        
        # Convert quaternion to yaw
        q = Quaternion(quat[3], quat[0], quat[1], quat[2])
        _, _, self.odom_yaw = q.to_euler()
    
    def unpause(self):
        """Unpause Gazebo simulation"""
        try:
            self.unpause_srv()
        except:
            pass
    
    def pause(self):
        """Pause Gazebo simulation"""
        try:
            self.pause_srv()
        except:
            pass
    
    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.02)
    
    def teleport_robot(self, x, y, yaw):
        print(f"   📍 Teleporting to ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")
        
        # 1. Stop all movement commands
        self.stop_robot()
        
        # 2. Pause physics to "freeze" the robot parts together
        self.pause()
        rospy.sleep(0.1)
        
        quat = Quaternion.from_euler(0, 0, yaw)
        state = ModelState()
        state.model_name = self.robot_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.05 # Lift slightly off ground to prevent clipping
        state.pose.orientation.x = quat.x
        state.pose.orientation.y = quat.y
        state.pose.orientation.z = quat.z
        state.pose.orientation.w = quat.w
        
        # CRITICAL: Reset all velocities to 0 so it doesn't "slide" after teleport
        state.twist.linear.x = 0; state.twist.linear.y = 0; state.twist.linear.z = 0
        state.twist.angular.x = 0; state.twist.angular.y = 0; state.twist.angular.z = 0
        state.reference_frame = "world"
        
        if self.set_model_state_srv:
            self.set_model_state_srv(state)
        
        # 3. Wait while paused so TF can catch up
        rospy.sleep(0.2)
        self.unpause()
    
    def reset_path_tracking(self):
        """Reset path tracking for new trial"""
        self._update_odom_from_broadcaster()
        self.path_points = [(self.odom_x, self.odom_y)]
        self.path_length = 0.0
        self.last_x = self.odom_x
        self.last_y = self.odom_y
        self.reset_collision_state()

    def reset_collision_state(self):
        """Reset collision detection state for new trial"""
        self.is_collision = False
    
    def update_path_tracking(self):
        """Update path tracking"""
        self._update_odom_from_broadcaster()
        
        dx = self.odom_x - self.last_x
        dy = self.odom_y - self.last_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if 0.01 < dist < 0.5:  # Reasonable movement
            self.path_length += dist
            self.path_points.append((self.odom_x, self.odom_y))
        
        self.last_x = self.odom_x
        self.last_y = self.odom_y
    
    def get_random_spawn_position(self):
        """Get random valid spawn position"""
        for _ in range(100):
                    # while not x_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            yaw = np.random.uniform(-math.pi, math.pi)
            
            if self._is_valid_position(x, y):
                return x, y, yaw
        
        return 0.0, 0.0, 0.0
    
    def _is_valid_position(self, x, y):
        """Check if position is valid"""
        goal_ok = True

        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            goal_ok = False

        if -1.3 > x > -2.7 and 4.7 > y > -0.2:
            goal_ok = False

        if -0.3 > x > -4.2 and 2.7 > y > 1.3:
            goal_ok = False

        if -0.8 > x > -4.2 and -2.3 > y > -4.2:
            goal_ok = False

        if -1.3 > x > -3.7 and -0.8 > y > -2.7:
            goal_ok = False

        if 4.2 > x > 0.8 and -1.8 > y > -3.2:
            goal_ok = False

        if 4 > x > 2.5 and 0.7 > y > -3.2:
            goal_ok = False

        if 6.2 > x > 3.8 and -3.3 > y > -4.2:
            goal_ok = False

        if 4.2 > x > 1.3 and 3.7 > y > 1.5:
            goal_ok = False

        if -3.0 > x > -7.2 and 0.5 > y > -1.5:
            goal_ok = False

        if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
            goal_ok = False

        return goal_ok
    
    def get_distance_to_goal(self, goal_x, goal_y):
        """Get current distance to goal"""
        self._update_odom_from_broadcaster()
        return math.sqrt((self.odom_x - goal_x)**2 + (self.odom_y - goal_y)**2)
    
    def shutdown(self):
        """Clean shutdown"""
        self.stop_robot()
        self.pause()
        if hasattr(self, 'odom_broadcaster'):
            self.odom_broadcaster.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# MOVE BASE LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
class MoveBaseLauncher:
    """Launches and manages move_base"""
    
    def __init__(self):
        self.processes = []
        self.client = None
        self.clear_costmaps_srv = None
    
    def launch(self):
        rospy.loginfo("🚀 Launching move_base...")
        
        # Kill the "thief" nodes that are stealing TF authority
        # subprocess.run(["rosnode", "kill", "/odom_to_base_link"], stderr=subprocess.DEVNULL)
        # subprocess.run(["rosnode", "kill", "/map_to_odom"], stderr=subprocess.DEVNULL)
        # subprocess.run(["rosnode", "kill", "/static_transform_publisher_1771571396919006725"], stderr=subprocess.DEVNULL)
        # subprocess.run(["rosnode", "kill", "/static_transform_publisher_1771571991698390307"])
        # ONLY publish map->odom if gmapping isn't running
        # Based on your debug, gmapping IS running, so we comment this out or check:
        p1 = subprocess.Popen([
            "rosrun", "tf", "static_transform_publisher",
            "0", "0", "0", "0", "0", "0", "map", "odom", "100"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p1)

        time.sleep(1)
        
        # Launch move_base
        move_base_cmd = [
            "rosrun", "move_base", "move_base",
            "odom:=/r1/odom_custom",
            "cmd_vel:=/r1/cmd_vel"
            "scan:=/r1/front_laser/scan",
            "_global_costmap/global_frame:=map",
            "_global_costmap/robot_base_frame:=base_link",
            "_global_costmap/static_map:=false",
            "_global_costmap/rolling_window:=true",
            "_global_costmap/width:=20.0",
            "_global_costmap/height:=20.0",
            "_global_costmap/resolution:=0.1",
            "_global_costmap/update_frequency:=3.0",
            "_global_costmap/publish_frequency:=1.0",
            "_global_costmap/transform_tolerance:=1.0",
            "_global_costmap/obstacle_layer/observation_sources:=scan",
            "_global_costmap/obstacle_layer/scan/topic:=/r1/front_laser/scan",
            "_global_costmap/obstacle_layer/scan/data_type:=LaserScan",
            "_global_costmap/obstacle_layer/scan/sensor_frame:=front_laser",
            "_global_costmap/obstacle_layer/scan/marking:=true",
            "_global_costmap/obstacle_layer/scan/clearing:=true",
            "_global_costmap/obstacle_layer/scan/inf_is_valid:=true",
            "_local_costmap/global_frame:=odom",
            "_local_costmap/robot_base_frame:=base_link",
            "_local_costmap/static_map:=false",
            "_local_costmap/rolling_window:=true",
            "_local_costmap/width:=5.0",
            "_local_costmap/height:=5.0",
            "_local_costmap/resolution:=0.05",
            "_local_costmap/update_frequency:=5.0",
            "_local_costmap/publish_frequency:=2.0",
            "_local_costmap/transform_tolerance:=1.0",
            "_local_costmap/obstacle_layer/observation_sources:=scan",
            "_local_costmap/obstacle_layer/scan/topic:=/r1/front_laser/scan",
            "_local_costmap/obstacle_layer/scan/data_type:=LaserScan",
            "_local_costmap/obstacle_layer/scan/sensor_frame:=front_laser",
            "_local_costmap/obstacle_layer/scan/marking:=true",
            "_local_costmap/obstacle_layer/scan/clearing:=true",
            "_local_costmap/obstacle_layer/scan/inf_is_valid:=true",
            "_base_local_planner:=dwa_local_planner/DWAPlannerROS",
            "_base_global_planner:=navfn/NavfnROS",
            "_DWAPlannerROS/max_vel_x:=0.4",
            "_DWAPlannerROS/min_vel_x:=0.05",
            "_DWAPlannerROS/max_vel_y:=0.0"
            "_DWAPlannerROS/min_vel_y:=0.0",
            "_DWAPlannerROS/min_vel_trans:=0.05",
            "_DWAPlannerROS/max_vel_theta:=1.0",
            "_DWAPlannerROS/acc_lim_x:=2.5",
            "_DWAPlannerROS/acc_lim_theta:=3.2",
            "_DWAPlannerROS/xy_goal_tolerance:=0.5",
            "_DWAPlannerROS/yaw_goal_tolerance:=0.5",
            "_local_costmap/obstacle_layer/scan/sensor_frame:=front_laser",
            "_local_costmap/obstacle_layer/scan/topic:=/r1/front_laser/scan"
        ]
        
        p2 = subprocess.Popen(move_base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p2)
        rospy.loginfo("  ✓ move_base started")
        
        # Launch RViz
        # p3 = subprocess.Popen(["rosrun", "rviz", "rviz"], 
        #                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # self.processes.append(p3)
        # rospy.loginfo("  ✓ RViz started")
        
        # Wait for action server
        rospy.loginfo("  ⏳ Waiting for move_base...")
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        
        if not self.client.wait_for_server(timeout=rospy.Duration(30)):
            rospy.logerr("  ❌ move_base timeout!")
            return False
        
        rospy.loginfo("  ✅ move_base ready!")
        
        # Setup clear costmaps service
        try:
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=5)
            self.clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        except:
            pass
        
        time.sleep(2.0)
        return True
    
    def clear_costmaps(self):
        """Clear move_base costmaps"""
        if self.clear_costmaps_srv:
            try:
                self.clear_costmaps_srv()
            except:
                pass
    
    def cancel_goals(self):
        """Cancel all navigation goals"""
        if self.client:
            try:
                self.client.cancel_all_goals()
                rospy.sleep(0.2)
            except:
                pass
    
    def shutdown(self):
        """Shutdown move_base"""
        self.cancel_goals()
        for p in self.processes:
            try:
                p.terminate()
                p.wait(timeout=2)
            except:
                try:
                    p.kill()
                except:
                    pass
        self.processes = []


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATORS
# ═══════════════════════════════════════════════════════════════════════════════
class MoveBaseNavigator:
    """Navigation using move_base"""
    
    def __init__(self, gazebo: GazeboInterface, move_base: MoveBaseLauncher):
        self.gazebo = gazebo
        self.move_base = move_base
        self.client = move_base.client
    
    def navigate_to(self, goal_x, goal_y):
        """Navigate to goal. Returns (success, elapsed_time)"""
        if not self.client:
            return False, 0.0
        
        # Create and send goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_x
        goal.target_pose.pose.position.y = goal_y
        
        self.gazebo._update_odom_from_broadcaster()
        yaw = math.atan2(goal_y - self.gazebo.odom_y, goal_x - self.gazebo.odom_x)
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2)
        
        self.client.send_goal(goal)
        rospy.loginfo(f"  🎯 Goal sent: ({goal_x:.2f}, {goal_y:.2f})")
        
        start_time = time.time()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            state = self.client.get_state()
            elapsed = time.time() - start_time
            rospy.logwarn(f" AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
            # Update path tracking
            self.gazebo.update_path_tracking()
            
            dist = self.gazebo.get_distance_to_goal(goal_x, goal_y)
            
            # Success
            if dist < SUCCESS_RADIUS:
                self.client.cancel_goal()
                return True, elapsed
            
            if self.gazebo.check_collision():
                rospy.logwarn(f"  🚨 COLLISION! Min distance: {COLLISION_DISTANCE}m")
                self.client.cancel_goal()
                self.gazebo.stop_robot()
                return False, elapsed  # Failed due to collision
            
            
            # Failures
            if elapsed > 3.0 and state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                return False, elapsed
            
            # Timeout
            if elapsed > GOAL_TIMEOUT:
                self.client.cancel_goal()
                return False, elapsed
            
            rate.sleep()
        
        return False, time.time() - start_time


class APFNavigator:
    """Artificial Potential Field navigation"""
    
    K_ATT = 1.5
    K_REP = 0.8
    D0 = 1.2
    MAX_LIN = 0.4
    MAX_ANG = 1.2
    
    def __init__(self, gazebo: GazeboInterface):
        self.gazebo = gazebo
        self.cmd_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
    
    def navigate_to(self, goal_x, goal_y):
        """Navigate using APF"""
        start_time = time.time()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            
            # Update position
            self.gazebo.update_path_tracking()
            
            dist = self.gazebo.get_distance_to_goal(goal_x, goal_y)
            if dist < SUCCESS_RADIUS:
                self._stop()
                return True, elapsed
            if self.gazebo.check_collision():
                rospy.logwarn(f"  🚨 COLLISION! Min distance: {COLLISION_DISTANCE}m")
                self._stop()
                return False, elapsed
            if elapsed > GOAL_TIMEOUT:
                self._stop()
                return False, elapsed
            
            cmd = self._compute_apf(goal_x, goal_y)
            self.cmd_pub.publish(cmd)
            
            rate.sleep()
        
        self._stop()
        return False, time.time() - start_time
    
    def _compute_apf(self, goal_x, goal_y):
        """Compute APF velocity command"""
        self.gazebo._update_odom_from_broadcaster()
        rx, ry, ryaw = self.gazebo.odom_x, self.gazebo.odom_y, self.gazebo.odom_yaw
        
        # Attractive force
        dx, dy = goal_x - rx, goal_y - ry
        f_att_x = self.K_ATT * dx
        f_att_y = self.K_ATT * dy
        
        # Repulsive force
        f_rep_x, f_rep_y = 0.0, 0.0
        
        for i, r in enumerate(self.gazebo.laser_ranges):
            if r < 0.15 or not math.isfinite(r) or r > self.D0:
                continue
            
            angle = self.gazebo.laser_angle_min + i * self.gazebo.laser_angle_increment + ryaw
            ox = rx + r * math.cos(angle)
            oy = ry + r * math.sin(angle)
            
            dox, doy = rx - ox, ry - oy
            d = math.sqrt(dox*dox + doy*doy) + 1e-6
            mag = self.K_REP * (1.0/d - 1.0/self.D0) / (d*d)
            
            f_rep_x += mag * dox / d
            f_rep_y += mag * doy / d
        
        fx = f_att_x + f_rep_x
        fy = f_att_y + f_rep_y
        
        cmd = Twist()
        cmd.linear.x = max(-self.MAX_LIN, min(self.MAX_LIN,
                          fx*math.cos(ryaw) + fy*math.sin(ryaw)))
        cmd.angular.z = max(-self.MAX_ANG, min(self.MAX_ANG,
                           -fx*math.sin(ryaw) + fy*math.cos(ryaw)))
        
        return cmd
    
    def _stop(self):
        self.cmd_pub.publish(Twist())


class Bug2Navigator:
    """Bug2 algorithm navigation"""
    
    LIN = 0.25
    ANG = 0.8
    OBS_DIST = 0.45
    WALL_DIST = 0.35
    
    def __init__(self, gazebo: GazeboInterface):
        self.gazebo = gazebo
        self.cmd_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.state = "GO_TO_GOAL"
        self.leave_dist = None
    
    def navigate_to(self, goal_x, goal_y):
        """Navigate using Bug2"""
        self.state = "GO_TO_GOAL"
        self.leave_dist = None
        
        start_time = time.time()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            
            self.gazebo.update_path_tracking()
            
            dist = self.gazebo.get_distance_to_goal(goal_x, goal_y)
            if dist < SUCCESS_RADIUS:
                self._stop()
                return True, elapsed
            
            if elapsed > GOAL_TIMEOUT:
                self._stop()
                return False, elapsed
            
            cmd = self._compute_bug2(goal_x, goal_y, dist)
            self.cmd_pub.publish(cmd)
            
            rate.sleep()
        
        self._stop()
        return False, time.time() - start_time
    
    def _compute_bug2(self, goal_x, goal_y, dist):
        """Compute Bug2 velocity command"""
        ranges = self.gazebo.laser_ranges
        n = len(ranges)
        
        if n == 0:
            return Twist()
        
        def get_sector(start_deg, end_deg):
            s = int((start_deg + 180) / 360 * n) % n
            e = int((end_deg + 180) / 360 * n) % n
            idx = range(s, e) if s < e else list(range(s, n)) + list(range(0, e))
            vals = [ranges[i] for i in idx if math.isfinite(ranges[i]) and ranges[i] > 0.05]
            return min(vals) if vals else float('inf')
        
        front = get_sector(-30, 30)
        left = get_sector(60, 120)
        
        self.gazebo._update_odom_from_broadcaster()
        rx, ry, ryaw = self.gazebo.odom_x, self.gazebo.odom_y, self.gazebo.odom_yaw
        cmd = Twist()
        
        if self.state == "GO_TO_GOAL":
            if front < self.OBS_DIST:
                self.leave_dist = dist
                self.state = "FOLLOW_WALL"
            else:
                err = math.atan2(goal_y - ry, goal_x - rx) - ryaw
                while err > math.pi: err -= 2*math.pi
                while err < -math.pi: err += 2*math.pi
                
                cmd.linear.x = self.LIN if abs(err) < 0.15 else 0.0
                cmd.angular.z = self.ANG * np.sign(err) if abs(err) > 0.15 else 0.8*err
        
        else:  # FOLLOW_WALL
            if self.leave_dist and dist < self.leave_dist - 0.2:
                self.state = "GO_TO_GOAL"
            elif front < self.OBS_DIST:
                cmd.angular.z = -self.ANG
            elif left > self.WALL_DIST * 1.5:
                cmd.linear.x = self.LIN * 0.5
                cmd.angular.z = self.ANG * 0.5
            else:
                cmd.linear.x = self.LIN
                cmd.angular.z = -0.8 * (self.WALL_DIST - left)
        
        return cmd
    
    def _stop(self):
        self.cmd_pub.publish(Twist())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
class SOTARunner:
    """Main SOTA navigation runner"""
    
    def __init__(self, algorithm: str = "dwa"):
        self.algorithm = algorithm.lower()
        
        # Initialize Gazebo (includes OdometryBroadcaster)
        self.gazebo = GazeboInterface()
        
        # Initialize move_base if needed
        self.move_base = None
        if self.algorithm in NEEDS_MOVE_BASE:
            self.move_base = MoveBaseLauncher()
            if not self.move_base.launch():
                rospy.logwarn("move_base failed, falling back to APF")
                self.algorithm = "apf"
        
        # Create navigator
        if self.algorithm in NEEDS_MOVE_BASE and self.move_base:
            self.navigator = MoveBaseNavigator(self.gazebo, self.move_base)
        elif self.algorithm == "apf":
            self.navigator = APFNavigator(self.gazebo)
        elif self.algorithm == "bug2":
            self.navigator = Bug2Navigator(self.gazebo)
        else:
            self.navigator = APFNavigator(self.gazebo)
        
        print(f"✅ SOTARunner ready - {self.algorithm.upper()}")
    
    def run_trials(self, goal_x, goal_y, object_name="goal", num_trials=TRIALS_PER_GOAL):
        """Run navigation trials"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"🎯 Goal: ({goal_x:.2f}, {goal_y:.2f}) - {object_name}")
        print(f"{'='*60}")
        
        for trial in range(num_trials):
            result = self._run_single_trial(goal_x, goal_y, object_name, trial + 1, num_trials)
            results.append(result)
        
        self._save_results(results)
        
        successes = sum(1 for r in results if r['success'])
        print(f"\n📊 Summary: {successes}/{num_trials} successful ({100*successes/num_trials:.1f}%)")
        
        return results
    
    def _run_single_trial(self, goal_x, goal_y, object_name, trial_num, total_trials):
        """Run a single navigation trial"""
        print(f"\n🚀 Trial {trial_num}/{total_trials}")
        ground_true_table = pd.read_csv('_models.csv')
        coordinates = ground_true_table.loc[ground_true_table['Model Name'] == object_name, ['X', 'Y', 'Z']]
        true_x, true_y = coordinates['X'].iloc[0], coordinates['Y'].iloc[0]
        
        # Cancel active goals
        if self.move_base:
            self.move_base.cancel_goals()
        
        # Pause simulation
        self.gazebo.pause()
        rospy.sleep(0.2)
        
        # Teleport robot to random start
        start_x, start_y, start_yaw = self.gazebo.get_random_spawn_position()
        self.gazebo.teleport_robot(start_x, start_y, start_yaw)
        
        # Unpause and wait for sensors
        self.gazebo.unpause()
        rospy.sleep(1.5)
        
        # Clear costmaps
        if self.move_base:
            self.move_base.clear_costmaps()
            rospy.sleep(0.5)
        
        # Reset path tracking
        self.gazebo.reset_path_tracking()
        
        self.gazebo._update_odom_from_broadcaster()
        start_pos = (self.gazebo.odom_x, self.gazebo.odom_y)
        print(f"   Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        
        # Navigate!
        success, nav_time = self.navigator.navigate_to(goal_x, goal_y)




        # Collect results
        self.gazebo._update_odom_from_broadcaster()
        final_pos = (self.gazebo.odom_x, self.gazebo.odom_y)
        final_distance = float(np.sqrt((final_pos[0] - true_x)**2 + (final_pos[1] - true_y)**2))

        success = False 
        
        # Determine success - info is a boolean in your environment
        distance_threshold = 1.00 # meters
        if final_distance <= distance_threshold:
            success = True

        # Compute SPL
        start_dist = math.sqrt((start_pos[0] - goal_x)**2 + (start_pos[1] - goal_y)**2)
        path_length = max(self.gazebo.path_length, 0.01)
        spl = (1.0 if success else 0.0) * (start_dist / max(path_length, start_dist))
        
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {status} | Path: {path_length:.2f}m | Time: {nav_time:.1f}s | Dist: {final_distance:.2f}m")
        
        return {
            'algorithm': self.algorithm.upper(),
            'object_name': object_name,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'start_x': start_pos[0],
            'start_y': start_pos[1],
            'final_x': final_pos[0],
            'final_y': final_pos[1],
            'true_x': true_x,
            'true_y': true_y,
            'success': success,
            'final_distance': final_distance,
            'path_length': path_length,
            'time': nav_time,
            'spl': spl,
            'trial': trial_num,
        }
    
    def _save_results(self, results):
        """Save results to Excel"""
        df_new = pd.DataFrame(results)
        
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_excel(EXCEL_FILE, index=False)
        print(f"\n📁 Results saved to {EXCEL_FILE}")

    def get_robot_position(self):
        """Get current robot position (same as RobotRunner)"""
        self.gazebo._update_odom_from_broadcaster()
        return (self.gazebo.odom_x, self.gazebo.odom_y)
    
    def sendGoal(self, goal_x, goal_y, true_x, true_y, object_name):
        """
        Navigate to goal position and track metrics.
        Same interface as RobotRunner.sendGoal()
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"🎯 Object: {object_name}")
        print(f"   Goal: ({goal_x:.2f}, {goal_y:.2f})")
        print(f"   True: ({true_x:.2f}, {true_y:.2f})")
        print(f"{'='*60}")
        
        for i in range(10):
            print(f"\n🚀 [{self.algorithm.upper()}] Trial {i+1}/{10}")
            
            # Cancel active goals
            if self.move_base:
                self.move_base.cancel_goals()
            
            # Pause simulation
            self.gazebo.pause()
            rospy.sleep(0.2)
            
            # Teleport to random start
            start_x, start_y, start_yaw = self.gazebo.get_random_spawn_position()
            self.gazebo.teleport_robot(start_x, start_y, start_yaw)
            print(f"   📍 Start: ({start_x:.2f}, {start_y:.2f})")
            
            # Unpause and wait for sensors
            self.gazebo.unpause()
            rospy.sleep(1.5)
            
            # Clear costmaps
            if self.move_base:
                self.move_base.clear_costmaps()
                rospy.sleep(0.5)
            
            # Reset tracking
            self.gazebo.reset_path_tracking()
            self.start_pos = self.get_robot_position()
            
            # Navigate
            success, nav_time = self.navigator.navigate_to(goal_x, goal_y)
            self.gazebo.stop_robot()
    
            if self.move_base:
                self.move_base.cancel_goals()
                
            # Collect results
            final_pos = self.get_robot_position()
            final_distance = float(np.sqrt(
                (final_pos[0] - true_x)**2 + 
                (final_pos[1] - true_y)**2
            ))
            
            # Check success using distance threshold (same as RobotRunner)
            # distance_threshold = SUCCESS_RADIUS
            # success = final_distance <= distance_threshold
            
            path_length = self.gazebo.path_length
            total_time = self.gazebo.total_time
            
            print(f"   {'✅ Success' if success else '❌ Failed'} | "
                  f"Path: {path_length:.2f}m | Time: {total_time:.1f}s | "
                  f"Final distance: {final_distance:.2f}m")
            
            # Store result (same format as RobotRunner)
            results.append({
                'goal_x': goal_x,
                'goal_y': goal_y,
                'success': success,
                'reward': 0,  # No reward for classical methods
                'path_length': path_length,
                'steps': int(total_time * 10),  # Approximate steps
                'final_distance': final_distance,
                'start_x': self.start_pos[0],
                'start_y': self.start_pos[1],
                'final_x': final_pos[0],
                'final_y': final_pos[1],
                'object_name': object_name,
                'true_x': true_x,
                'true_y': true_y,
                'taken_time': nav_time,
                'algorithm': self.algorithm.upper(),
            })
        
        # Save to Excel (same as RobotRunner)
        df_new = pd.DataFrame(results)
        
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_excel(EXCEL_FILE, index=False)
        print(f"\n📁 Excel updated → {EXCEL_FILE} ({len(df_new)} new rows added)\n")
        
        return results

    def shutdown(self):
        """Clean shutdown"""
        if self.move_base:
            self.move_base.shutdown()
        self.gazebo.shutdown()

def load_objects_from_jsonl(filepath):
    """Load detected objects from JSONL file"""
    objects = []
    
    if not os.path.exists(filepath):
        print(f"❌ Object database not found: {filepath}")
        return objects
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Check required fields
                if 'world_x' in obj and 'world_y' in obj:
                    objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse line: {e}")
                continue
    
    print(f"✅ Loaded {len(objects)} objects from {filepath}")
    return objects
def deduplicate_objects(objects, distance_threshold=1.0):
    """Remove duplicate objects (same name, close position)"""
    unique = []
    
    for obj in objects:
        name = obj.get('object_class', 'unknown')
        x = obj['world_x']
        y = obj['world_y']
        
        # Check if similar object already exists
        is_duplicate = False
        for u in unique:
            u_name = u.get('object_name', u.get('name', 'unknown'))
            if name.lower() == u_name.lower():
                dist = ((x - u['world_x'])**2 + (y - u['world_y'])**2)**0.5
                if dist < distance_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique.append(obj)
    
    print(f"📋 {len(unique)} unique objects after deduplication")
    return unique

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SOTA Navigation Runner")
    parser.add_argument("--algorithm", "-a", type=str, default="dwa",
                       choices=["dwa", "teb", "navfn", "apf", "bug2", "all"],
                       help="Navigation algorithm (use 'all' to test all)")
    parser.add_argument("--objects_file", "-f", type=str, default=OBJECTS_JSONL,
                       help="Path to objects JSONL file")
    parser.add_argument("--trials", "-n", type=int, default=10,
                       help="Number of trials per object")
    parser.add_argument("--output", "-o", type=str, default=EXCEL_FILE,
                       help="Output Excel file")
    parser.add_argument("--goal_x", "-x", type=float, default=None,
                       help="Single goal X (overrides JSONL)")
    parser.add_argument("--goal_y", "-y", type=float, default=None,
                       help="Single goal Y (overrides JSONL)")
    
    args = parser.parse_args()
    
    # Update globals
    
    # Determine algorithms to test
    if args.algorithm == "all":
        algorithms = ["apf", "bug2", "dwa"]
    else:
        algorithms = [args.algorithm]
    
    runner = None
    all_results = []
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # OPTION 1: Single goal from command line
        # ═══════════════════════════════════════════════════════════════════
        if args.goal_x is not None and args.goal_y is not None:
            print(f"\n🎯 Testing single goal: ({args.goal_x}, {args.goal_y})")
            
            for algo in algorithms:
                if runner:
                    runner.shutdown()
                    time.sleep(2)
                
                runner = SOTARunner(algorithm=algo)
                results = runner.sendGoal(
                    goal_x=args.goal_x,
                    goal_y=args.goal_y,
                    true_x=args.goal_x,
                    true_y=args.goal_y,
                    object_name="manual_goal"
                )
                all_results.extend(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # OPTION 2: Load objects from JSONL file
        # ═══════════════════════════════════════════════════════════════════
        else:
            print(f"\n📂 Loading objects from: {args.objects_file}")
            
            objects = load_objects_from_jsonl(args.objects_file)
            
            if not objects:
                print("❌ No objects found. Use --goal_x and --goal_y for manual testing.")
                exit(1)
            
            # Deduplicate
            objects = deduplicate_objects(objects)
            
            # Print objects
            print(f"\n📋 Objects to navigate ({len(objects)}):")
            for i, obj in enumerate(objects, 1):
                name = obj.get('object_class',  'unknown')
                print(f"   {i}. {name}: ({obj['world_x']:.2f}, {obj['world_y']:.2f})")
            
            print(f"\n🤖 Algorithms: {', '.join(a.upper() for a in algorithms)}")
            print(f"🔄 Trials per object: {args.trials}")
            print(f"📊 Total trials: {len(objects) * len(algorithms) * args.trials}")
            
            # Run trials for each algorithm and object
            for algo in algorithms:
                print(f"\n{'='*70}")
                print(f"🤖 TESTING ALGORITHM: {algo.upper()}")
                print(f"{'='*70}")
                
                if runner:
                    runner.shutdown()
                    time.sleep(2)
                
                runner = SOTARunner(algorithm=algo)
                
                for obj_idx, obj in enumerate(objects, 1):
                    object_name = obj.get('object_class',  'unknown')
                    ground_true_table = pd.read_csv('_models.csv')
                    coordinates = ground_true_table.loc[ground_true_table['Model Name'] == object_name, ['X', 'Y', 'Z']]
                    true_x, true_y = coordinates['X'].iloc[0], coordinates['Y'].iloc[0]
                    world_x = obj['world_x']
                    world_y = obj['world_y']

                    
                    print(f"\n{'─'*60}")
                    print(f"📍 Object {obj_idx}/{len(objects)}: {object_name}")
                    print(f"{'─'*60}")
                    
                    results = runner.run_trials(
                        goal_x=world_x,
                        goal_y=world_y,
                        object_name=object_name
                    )
                    all_results.extend(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRINT SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        # if all_results:
        #     print(f"\n{'='*70}")
        #     print("📊 FINAL SUMMARY")
        #     print(f"{'='*70}")
            
        #     df = pd.DataFrame(all_results)
            
        #     for algo in algorithms:
        #         algo_data = df[df['algorithm'] == algo.upper()]
        #         if len(algo_data) > 0:
        #             sr = algo_data['success'].mean() * 100
        #             avg_path = algo_data['path_length'].mean()
        #             avg_time = algo_data['taken_time'].mean()
                    
        #             print(f"{algo.upper():>10}: SR={sr:.1f}%, "
        #                   f"Avg Path={avg_path:.2f}m, "
        #                   f"Avg Time={avg_time:.1f}s")
            
        #     print(f"\n📁 All results saved to: {EXCEL_FILE}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if runner:
            runner.shutdown()

