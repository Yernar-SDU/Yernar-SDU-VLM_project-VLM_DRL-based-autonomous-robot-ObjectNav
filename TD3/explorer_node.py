#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
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
from pixel_to_cords import VLM_Response_Processor


class RobotExplorer:

    def __init__(self, vlmType='moondream', world='TD3_signs2.world'):
        self.vlm_type = vlmType

        # ── Launch ROS infrastructure ─────────────────────────────────────
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        time.sleep(2.0)

        script_dir  = os.path.dirname(os.path.realpath(__file__))
        launch_file = os.path.join(script_dir, "assets", "multi_robot_scenario.launch")
        if not os.path.exists(launch_file):
            raise IOError(f"Launch file not found: {launch_file}")

        print(f"Launching world: {world}")
        subprocess.Popen(["roslaunch", "-p", port, launch_file, f"world_name:={world}"])
        print("LiDAR system launched!")
        time.sleep(4.0)

        rospy.init_node('robot_explorer', anonymous=False)

        # ── Topics ────────────────────────────────────────────────────────
        self.camera_topic     = "/realsense_camera/color/image_raw"
        self.pointcloud_topic = "/realsense_camera/depth/color/points"
        self.map_topic        = "/map"
        self.odom_topic       = "/r1/odom"
        self.save_dir         = "/tmp/exploration_data"
        os.makedirs(self.save_dir, exist_ok=True)

        # ── Helpers ───────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.vlm_detector = VLM_Response_Processor(self.vlm_type, save_dir=self.save_dir)

        # ── VLM timing ────────────────────────────────────────────────────
        self.last_object_detection_time = rospy.Time.now().to_sec()
        self.object_detection_interval  = 2  # seconds

        # ── Metadata file ─────────────────────────────────────────────────
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        if not os.path.exists(self.metadata_file):
            open(self.metadata_file, "w").close()

        # ── Sensor state ──────────────────────────────────────────────────
        self.current_pose = None
        self.frame_id     = 0
        self.map_data     = None
        self.map_info     = None

        # ── Navigation state ──────────────────────────────────────────────
        self.goal_reached        = True
        self.stuck_counter       = 0
        self.max_stuck_count     = 6

        # 40 entries x 1.5s timer = 60s of history
        self.position_history       = deque(maxlen=40)
        self.min_movement_threshold = 0.25  # metres over ~15s window

        # ── Blacklisting ──────────────────────────────────────────────────
        self.current_goal                  = None
        self.failed_goals                  = set()   # temporary
        self.permanent_blacklist           = set()   # survives recovery
        self.goal_attempt_count            = {}
        self.max_attempts_per_goal         = 5
        self.permanent_blacklist_threshold = 10
        self.blacklist_radius              = 0.35    # was 0.8

        # ── Goal rate-limiter ─────────────────────────────────────────────
        self.last_goal_send_time    = 0
        self.min_goal_send_interval = 3.0

        # ── Exploration timing ────────────────────────────────────────────
        self.exploration_active = False
        self.last_goal_time     = 0
        self.goal_timeout       = 45.0   # was 25s

        # ── Cycle / visited state ─────────────────────────────────────────
        self.recovery_mode          = False
        self.consecutive_same_goals = 0
        self.last_goal_coords       = None
        self.stop_condition         = False
        self.visited_positions      = deque(maxlen=100)

        # Cache rebuilt every 2s by _analyze_map_timer
        self._map_analysis_cache = ([], [])

        # ── Publishers ────────────────────────────────────────────────────
        self.goal_pub    = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=1)
        self.cancel_pub  = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber(self.map_topic,          OccupancyGrid,    self.map_callback)
        rospy.Subscriber(self.odom_topic,         Odometry,         self.odom_callback)
        rospy.Subscriber('/move_base/status',     GoalStatusArray,  self.goal_status_callback)

        image_sub = Subscriber(self.camera_topic,     Image)
        pc_sub    = Subscriber(self.pointcloud_topic, PointCloud2)
        odom_sub  = Subscriber(self.odom_topic,       Odometry)
        sync = ApproximateTimeSynchronizer(
            [image_sub, pc_sub, odom_sub], queue_size=100, slop=0.01)
        sync.registerCallback(self.synchronized_callback)

        rospy.loginfo("=" * 60)
        rospy.loginfo("ROBOT EXPLORER — STARTED")
        rospy.loginfo("=" * 60)

        # Decouple heavy map analysis (2s) from fast control loop (0.5s)
        rospy.Timer(rospy.Duration(1.5), self.position_check)
        rospy.Timer(rospy.Duration(3.0), self.cycle_detection)
        rospy.Timer(rospy.Duration(0.5), self.exploration_control)
        rospy.Timer(rospy.Duration(2.0), self._analyze_map_timer)

        rospy.spin()

    # ─────────────────────────────────────────────────────────────────────
    # Sensor callbacks
    # ─────────────────────────────────────────────────────────────────────

    def synchronized_callback(self, image_msg, pc_msg, odom_msg):
        try:
            t = rospy.Time.now().to_sec()
            if (int(t) % int(self.object_detection_interval) == 0 and
                    t - self.last_object_detection_time >= self.object_detection_interval):

                rospy.loginfo("⏸️  VLM detection starting...")
                t0 = rospy.Time.now().to_sec()

                def _wrapper(img, pc, pose, ts, start):
                    n    = self.vlm_detector.detect_objects(img, pc, pose, ts, self.frame_id)
                    diff = rospy.Time.now().to_sec() - start
                    with open("latencies.txt", "a") as f:
                        f.write(f"Time: {ts}, Latency: {diff:.4f}s, Count: {n}\n")
                    rospy.loginfo(f"✅ VLM: {n} objects in {diff:.2f}s")
                    self.save_image_with_metadata(img, ts, pose)

                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                threading.Thread(
                    target=_wrapper,
                    args=(cv_image, pc_msg, odom_msg.pose.pose, t, t0),
                    daemon=True
                ).start()
                self.last_object_detection_time = t

        except Exception as e:
            rospy.logerr(f"synchronized_callback: {e}")

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        grid = np.zeros_like(self.map_data, dtype=np.uint8)
        grid[self.map_data == -1] = 127
        grid[self.map_data ==  0] = 255
        cv2.imwrite(os.path.join(self.save_dir, "map_latest.png"), np.flipud(grid))

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        pos = (self.current_pose.position.x, self.current_pose.position.y)
        self.position_history.append((pos, rospy.Time.now().to_sec()))
        if (not self.visited_positions or
                math.hypot(pos[0] - self.visited_positions[-1][0],
                           pos[1] - self.visited_positions[-1][1]) > 0.5):
            self.visited_positions.append(pos)

    def goal_status_callback(self, msg):
        if not msg.status_list:
            return
        s = msg.status_list[-1].status
        if s == 3:
            rospy.loginfo("✅ Goal reached!")
            self.goal_reached = True
            self.stuck_counter = 0
            self.recovery_mode = False
            self.consecutive_same_goals = 0
        elif s in [4, 5]:
            rospy.logwarn(f"❌ Goal failed (status {s})")
            self._handle_failed_goal()

    # ─────────────────────────────────────────────────────────────────────
    # Goal management
    # ─────────────────────────────────────────────────────────────────────

    def _handle_failed_goal(self):
        self.stuck_counter += 1
        if self.current_goal:
            key   = self._key(self.current_goal)
            count = self.goal_attempt_count.get(key, 0) + 1
            self.goal_attempt_count[key] = count
            # Store ACTUAL coordinates (not rounded keys) so radius checks are accurate
            if count >= self.permanent_blacklist_threshold:
                self.permanent_blacklist.add(self.current_goal)
                rospy.logwarn(f"🔒 PERM BL: {self.current_goal} after {count} attempts")
            elif count >= self.max_attempts_per_goal:
                self.failed_goals.add(self.current_goal)
                rospy.logwarn(f"🚫 Temp BL: {self.current_goal} after {count} attempts")
            if len(self.failed_goals) > 12:
                for g in list(self.failed_goals)[:4]:
                    self.failed_goals.discard(g)
        self.goal_reached = True
        rospy.logwarn(f"Stuck {self.stuck_counter}/{self.max_stuck_count} | "
                      f"TBL={len(self.failed_goals)} PBL={len(self.permanent_blacklist)}")

    def _key(self, coords, p=1):
        x, y = coords
        return (round(x * p) / p, round(y * p) / p)

    def _is_blacklisted(self, coords):
        if not coords:
            return False
        gx, gy = coords
        # Compare against actual stored coordinates (not rounded keys) for correct radius checks
        for fk in list(self.failed_goals) + list(self.permanent_blacklist):
            if math.hypot(gx - fk[0], gy - fk[1]) < self.blacklist_radius:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Stuck / cycle detection
    # ─────────────────────────────────────────────────────────────────────

    def cycle_detection(self, event):
        if self.stop_condition:
            return
        if self.last_goal_coords == self.current_goal:
            self.consecutive_same_goals += 1
        else:
            self.consecutive_same_goals = 0
            self.last_goal_coords = self.current_goal
        if self.consecutive_same_goals >= 3:
            rospy.logwarn(f"🔄 Cycle: same goal {self.consecutive_same_goals}x")
            if self.current_goal:
                self.failed_goals.add(self.current_goal)  # store actual coords
            self.stuck_counter = self.max_stuck_count
            self.consecutive_same_goals = 0

    def position_check(self, event):
        if self.stop_condition or len(self.position_history) < 10:
            return
        recent = list(self.position_history)[-10:]
        total  = sum(math.hypot(recent[i][0][0] - recent[i-1][0][0],
                                recent[i][0][1] - recent[i-1][0][1])
                     for i in range(1, len(recent)))
        if total < self.min_movement_threshold and not self.goal_reached:
            rospy.logwarn(f"🐌 Stuck: {total:.3f}m in ~15s")
            self.stuck_counter += 1
            self.goal_reached   = True

    # ─────────────────────────────────────────────────────────────────────
    # Map analysis (runs every 2s, NOT inside the 0.5s control loop)
    # ─────────────────────────────────────────────────────────────────────

    def _analyze_map_timer(self, event):
        if self.map_data is None:
            return
        frontiers = self._find_frontiers()
        valid     = self._filter_frontiers(frontiers)
        self._map_analysis_cache = (frontiers, valid)

        known = int(np.sum(self.map_data >= 0))
        pct   = known / self.map_data.size * 100
        rospy.loginfo_throttle(15,
            f"📊 {pct:.1f}% explored | "
            f"{len(frontiers)} raw → {len(valid)} valid frontiers | "
            f"TBL={len(self.failed_goals)} PBL={len(self.permanent_blacklist)}")

        if known > 200 and valid and not self.exploration_active:
            rospy.loginfo("🚀 Exploration activated!")
            self.exploration_active = True

    # ─────────────────────────────────────────────────────────────────────
    # Frontier detection
    # ─────────────────────────────────────────────────────────────────────

    def _find_frontiers(self):
        if self.map_data is None:
            return []
        raw = []
        h, w = self.map_data.shape
        # step=1: full resolution (was step=2 which missed thin corridors)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.map_data[y, x] != 0:
                    continue
                if any(self.map_data[y + dy, x + dx] == -1
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                    wx = x * self.map_info.resolution + self.map_info.origin.position.x
                    wy = y * self.map_info.resolution + self.map_info.origin.position.y
                    raw.append((wx, wy))
        return self._cluster(raw, radius=0.5)

    def _cluster(self, pts, radius=0.5):
        """Merge frontier cells within radius into centroids."""
        if not pts:
            return []
        used = [False] * len(pts)
        out  = []
        for i, (px, py) in enumerate(pts):
            if used[i]:
                continue
            grp = [(px, py)]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j] and math.hypot(px - pts[j][0], py - pts[j][1]) < radius:
                    grp.append(pts[j])
                    used[j] = True
            out.append((sum(p[0] for p in grp) / len(grp),
                        sum(p[1] for p in grp) / len(grp)))
        return out

    def _obstacle_ratio(self, wx, wy, r=0.4):
        """Fraction of nearby cells that are occupied (high → phantom frontier)."""
        if self.map_data is None or self.map_info is None:
            return 0.0
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        ri = max(1, int(r / self.map_info.resolution))
        h, w = self.map_data.shape
        p = self.map_data[max(0, my-ri):min(h, my+ri+1),
                          max(0, mx-ri):min(w, mx+ri+1)]
        return 0.0 if p.size == 0 else int(np.sum(p > 0)) / p.size

    def _filter_frontiers(self, frontiers):
        if not frontiers or self.current_pose is None:
            return []
        rx, ry = self.current_pose.position.x, self.current_pose.position.y
        valid  = []
        for (fx, fy) in frontiers:
            dist = math.hypot(rx - fx, ry - fy)
            if dist < 0.2 or dist > 15.0:
                continue
            if self._is_blacklisted((fx, fy)):
                continue
            if self._obstacle_ratio(fx, fy) > 0.25:
                continue
            if any(math.hypot(fx - vx, fy - vy) < 0.3
                   for vx, vy in self.visited_positions):
                continue
            valid.append(((fx, fy), dist))
        valid.sort(key=lambda x: x[1])
        return [f[0] for f in valid]

    def _select_frontier(self, valid):
        if not valid or self.current_pose is None:
            return valid[0] if valid else None
        rx, ry = self.current_pose.position.x, self.current_pose.position.y
        q = self.current_pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        best, best_score = None, -999
        for (fx, fy) in valid:
            dist  = math.hypot(rx - fx, ry - fy)
            gain  = self._unknown_near(fx, fy)
            adiff = abs(math.atan2(fy - ry, fx - rx) - yaw)
            while adiff > math.pi:
                adiff = abs(adiff - 2 * math.pi)
            align = 1.0 - adiff / math.pi
            score = gain * 3.0 + align * 0.3 - dist * 0.15
            if score > best_score:
                best_score = score
                best = (fx, fy)

        # 25% random fallback AFTER full loop (was incorrectly inside loop)
        if random.random() < 0.25:
            return random.choice(valid)
        return best

    def _unknown_near(self, wx, wy, radius=2.5):
        if self.map_data is None or self.map_info is None:
            return 0
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        r  = int(radius / self.map_info.resolution)
        h, w = self.map_data.shape
        p = self.map_data[max(0,my-r):min(h,my+r), max(0,mx-r):min(w,mx+r)]
        return int(np.sum(p == -1))

    # ─────────────────────────────────────────────────────────────────────
    # Main control loop
    # ─────────────────────────────────────────────────────────────────────

    def exploration_control(self, event):
        if self.stop_condition:
            return
        if not self.exploration_active:
            rospy.loginfo_throttle(20, "⏳ Waiting for map...")
            return
        if self.current_pose is None:
            rospy.loginfo_throttle(10, "📍 Waiting for pose...")
            return
        if self.stuck_counter >= self.max_stuck_count:
            rospy.logwarn("🔄 Recovery triggered!")
            self._recovery()
            return

        now       = rospy.Time.now().to_sec()
        timed_out = (now - self.last_goal_time) > self.goal_timeout
        if timed_out:
            rospy.logwarn(f"⏰ Timeout after {self.goal_timeout}s")
            if self.current_goal:
                self.failed_goals.add(self.current_goal)  # store actual coords

        if self.goal_reached or timed_out:
            self._send_goal()

    def _send_goal(self):
        now = rospy.Time.now().to_sec()
        if now - self.last_goal_send_time < self.min_goal_send_interval:
            return

        _, valid = self._map_analysis_cache
        if not valid:
            rospy.loginfo("No valid frontiers — clearing temp BL")
            self.failed_goals.clear()
            return

        sel = self._select_frontier(valid)
        if not sel:
            return
        tx, ty = sel

        # Avoid resending identical goal
        if (self.current_goal and
                abs(tx - self.current_goal[0]) < 0.1 and
                abs(ty - self.current_goal[1]) < 0.1 and
                len(valid) > 1):
            rospy.logwarn("Same goal — forcing random pick")
            tx, ty = random.choice(valid)

        goal = PoseStamped()
        goal.header.stamp    = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = tx
        goal.pose.position.y = ty
        goal.pose.position.z = 0.0
        # Orient toward the goal so the robot doesn't spin to face East at every waypoint
        travel_yaw = math.atan2(ty - self.current_pose.position.y,
                                tx - self.current_pose.position.x)
        goal.pose.orientation.z = math.sin(travel_yaw / 2.0)
        goal.pose.orientation.w = math.cos(travel_yaw / 2.0)

        self.current_goal        = (tx, ty)
        self.last_goal_time      = now
        self.last_goal_send_time = now
        self.goal_reached        = False
        self.goal_pub.publish(goal)

        dist = math.hypot(self.current_pose.position.x - tx,
                          self.current_pose.position.y - ty)
        rospy.loginfo(f"🎯 GOAL ({tx:.2f},{ty:.2f}) dist={dist:.2f}m | "
                      f"valid={len(valid)} TBL={len(self.failed_goals)} "
                      f"PBL={len(self.permanent_blacklist)}")

    # ─────────────────────────────────────────────────────────────────────
    # Recovery
    # ─────────────────────────────────────────────────────────────────────

    def _recovery(self):
        rospy.loginfo("🆘 Recovery...")
        self.recovery_mode = True
        self._stop()
        rospy.sleep(0.8)

        rtype = self.stuck_counter % 4
        if rtype == 0:
            self._move(-0.3, 0.0, 2.5)
        elif rtype == 1:
            self._move(0.0, random.choice([-0.8, 0.8]), 2.0)
            rospy.sleep(0.3)
            self._move(0.3, 0.0, 1.5)
        elif rtype == 2:
            cmd = Twist()
            for i in range(25):
                cmd.linear.x  = 0.1 + i * 0.012
                cmd.angular.z = 0.5
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(0.1)
            self._stop()
        else:
            self._move(0.0, random.choice([-1.2, 1.2]), 2.5)

        cleared = len(self.failed_goals)
        self.failed_goals.clear()
        rospy.loginfo(f"🧹 Cleared {cleared} temp BL (perm={len(self.permanent_blacklist)})")

        self.stuck_counter          = 0
        self.goal_reached           = True
        self.recovery_mode          = False
        self.consecutive_same_goals = 0
        rospy.sleep(1.5)

    def _stop(self):
        cmd = Twist()
        for _ in range(8):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)

    def _move(self, lin, ang, dur):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        for _ in range(int(dur * 10)):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        self._stop()

    # ─────────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────────

    def save_image_with_metadata(self, cv_image, ts, pose):
        meta = {"timestamp": ts}
        if pose and self.map_info:
            rx, ry = pose.position.x, pose.position.y
            q = pose.orientation
            theta = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            meta.update({
                "robot_x": rx, "robot_y": ry, "robot_theta": theta,
                "map_origin_x": self.map_info.origin.position.x,
                "map_origin_y": self.map_info.origin.position.y,
                "map_resolution": self.map_info.resolution,
                "map_cell_x": int((rx - self.map_info.origin.position.x) / self.map_info.resolution),
                "map_cell_y": int((ry - self.map_info.origin.position.y) / self.map_info.resolution),
            })
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(meta) + "\n")
        self.frame_id += 1
        rospy.loginfo(f"💾 Frame {self.frame_id} @ t={ts:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vlm",   "-a", default="moondream", choices=["moondream", "gpt4o"])
    p.add_argument("--world", "-w", default="TD3_signs2.world")
    args = p.parse_args()
    try:
        RobotExplorer(args.vlm, args.world)
    except rospy.ROSInterruptException:
        pass