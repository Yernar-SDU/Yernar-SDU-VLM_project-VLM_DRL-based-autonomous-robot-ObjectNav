#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import cv2
import os
import numpy as np
import math
import json
import base64
from openai import OpenAI
from GPT_VLM import GPT_VLMDetector
from OWL_VLM import OwlVLMDetector
import csv
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
from MND_VLM import MoonVLMDetector




class VLM_Response_Processor:
    """
    Handles all VLM-based object detection and coordinate transformation
    """
    
    def __init__(self, vlmType, save_dir="/tmp/exploration_data"):
        self.save_dir = save_dir
        self.bridge = CvBridge()

        # Camera calibration
        self.camera_to_robot_translation = np.array([0.55, 0.02, 0.32])
        self.camera_to_robot_rotation = 0.0
        
        # Object tracking
        self.detected_objects = {}
        self.object_id = 0
        self.objects_file = os.path.join(self.save_dir, "detected_objects.jsonl")
        self.load_existing_objects()


        world_file = "/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch/TD3_signs2.world"
        output_csv = "_models.csv"

        data = self.extract_model_positions(world_file)
        
        # Write to CSV
        with open(output_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Model Name", "X", "Y", "Z"])
            writer.writerows(data)

        df = pd.read_csv(output_csv)

        self.model_names = df["Model Name"].tolist()

        # Create objects file if it doesn't exist
        if not os.path.exists(self.objects_file):
            open(self.objects_file, "w").close()

        self.gptDetector = GPT_VLMDetector()
        if vlmType == 'moondream':
            print('changed to moondream')
            self.moonDetector = MoonVLMDetector("http://localhost:8000")
        


    def extract_model_positions(self, world_path):
        tree = ET.parse(world_path)
        root = tree.getroot()

        positions = []
        seen_models = set()

        IGNORE_NAMES = {'box', 'wall', 'ground', 'floor', 'ceiling'}

        for model in root.iter('model'):
            name = model.get('name')

            if name in IGNORE_NAMES:
                continue

            if name in seen_models:
                continue

            static_tag = model.find('static')
            if static_tag is not None and static_tag.text.strip().lower() in ['true', '1']:
                continue

            pose_tag = model.find('pose')
            if pose_tag is None:
                continue

            x, y, z, roll, pitch, yaw = map(float, pose_tag.text.split())

            positions.append([name, x, y, z])
            seen_models.add(name)

        return positions

    def get_coords_from_bbox(self, bbox, pointcloud, current_pose):
        """
        Extract 3D coordinates from entire bounding box
        Uses closest valid depth inside box (better for thin objects)
        """

        if not pointcloud:
            return None

        x1, y1, x2, y2 = bbox

        width = pointcloud.width
        height = pointcloud.height

        samples = []

        # shrink box to reduce background influence
        margin = 0.18
        w = x2 - x1
        h = y2 - y1

        x1 = int(x1 + w * margin)
        x2 = int(x2 - w * margin)
        y1 = int(y1 + h * margin)
        y2 = int(y2 - h * margin)

        try:
            for u in range(x1, x2, 2):   # step 2 = faster
                for v in range(y1, y2, 2):

                    if 0 <= u < width and 0 <= v < height:

                        points = list(pc2.read_points(
                            pointcloud,
                            field_names=("x", "y", "z"),
                            skip_nans=True,
                            uvs=[(int(u), int(v))]
                        ))

                        if points:
                            x, y, z = points[0]

                            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                                samples.append((x, y, z))

            if not samples:
                return None

            samples_array = np.array(samples)

            distances = np.linalg.norm(samples_array, axis=1)
            best_idx = np.argmin(distances)

            cam_x, cam_y, cam_z = samples_array[best_idx]


            return self.transform_to_world_coords(cam_x, cam_y, cam_z, current_pose)

        except Exception as e:
            rospy.logerr(f"BBox point cloud extraction error: {e}")
            return None


    def detect_objects(self, cv_image, pointcloud, pose, timestamp, frame_id):
        """
        Main detection pipeline
        Returns: Number of objects detected
        """

        if pose is None or pointcloud is None:
            rospy.logwarn("Cannot detect objects: missing pose or pointcloud")
            return 0
    
        
        try:
            detected_objects = self.moonDetector.detect(cv_image, self.model_names)
            # detected_objects = self.gptDetector.detect_with_vlm(cv_image, self.model_names)

            if not detected_objects:
                rospy.loginfo("No objects detected")
                return 0
            
            # Save visualization
            self.save_detection_visualization(cv_image, detected_objects, frame_id)
            
            # Process detections
            saved_count = 0
            for obj in detected_objects:
                world_coords = self.get_coords_from_bbox(
                    obj['bbox'], pointcloud, pose
                )


                
                if world_coords:
                    dx = world_coords[0] - pose.position.x
                    dy = world_coords[1] - pose.position.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist <= 10.0:
                        self.save_object_detection(obj, world_coords, timestamp, pose, frame_id)
                        saved_count += 1
                    else:
                        rospy.logwarn(f"Rejecting {obj['class']} - too far: {dist:.2f}m")
            
            return saved_count
            
        except Exception as e:
            rospy.logerr(f"Detection error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return 0
   
    def get_coords_from_pointcloud(self, bbox, pointcloud, current_pose):
        if not pointcloud:
            return None

        px1, py1, px2, py2 = bbox
        width = pointcloud.width
        height = pointcloud.height

        # Sample exact center pixel of the bounding box
        center_u = int((px1 + px2) / 2)
        center_v = int((py1 + py2) / 2)

        # Clamp to image bounds
        center_u = max(0, min(width - 1, center_u))
        center_v = max(0, min(height - 1, center_v))

        try:
            points = list(pc2.read_points(
                pointcloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
                uvs=[(center_u, center_v)]
            ))

            if not points:
                rospy.logwarn(f"No valid point at exact pixel ({center_u}, {center_v})")
                return None

            x, y, z = points[0]

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                rospy.logwarn(f"Non-finite point at ({center_u}, {center_v})")
                return None

            return self.transform_to_world_coords(x, y, z, current_pose)

        except Exception as e:
            rospy.logerr(f"Point cloud extraction error: {e}")
            return None
        
    def transform_to_world_coords(self, cam_x, cam_y, cam_z, current_pose):
        if current_pose is None:
            return None
        try:
            robot_pos = current_pose.position
            robot_quat = current_pose.orientation
            siny_cosp = 2.0 * (robot_quat.w * robot_quat.z + robot_quat.x * robot_quat.y)
            cosy_cosp = 1.0 - 2.0 * (robot_quat.y * robot_quat.y + robot_quat.z * robot_quat.z)
            robot_yaw = math.atan2(siny_cosp, cosy_cosp)

            
            point_robot_frame = np.array([cam_z, -cam_x, 0.0])
           
            point_robot = point_robot_frame + self.camera_to_robot_translation

            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            offset_x = point_robot[0] * cos_yaw - point_robot[1] * sin_yaw
            offset_y = point_robot[0] * sin_yaw + point_robot[1] * cos_yaw
         

            world_x = max(-5.0, min(5.0, robot_pos.x + offset_x))
            world_y = max(-5.0, min(5.0, robot_pos.y + offset_y))
   

            return (world_x, world_y, 0.0)
        except Exception as e:
            rospy.logerr(f"Transform error: {e}")
            return None

    def cluster_observations(self, observations, cluster_radius=0.5):
        """
        Groups all stored observations for one object class into spatial clusters,
        then returns the mean position of the largest (most-observed) cluster.

        Example
        -------
        obs 1: (3.2, 1.0)   ──┐
        obs 2: (5.0, -2.0)    │  cluster A  (2 points, within 0.5 m of each other)
        obs 3: (3.3, 1.2)   ──┘
                            cluster B  (1 point – outlier)

        → picks cluster A → returns average (3.25, 1.1)

        Parameters
        ----------
        observations  : list of dicts with keys "world_x", "world_y", "world_z"
        cluster_radius: two points belong to the same cluster if their distance
                        is ≤ this value (metres).  Tune to your environment.

        Returns
        -------
        (best_x, best_y, best_z, cluster_size)  or  None if observations is empty.
        """
        if not observations:
            return None

        clusters = []   # list-of-lists, each inner list is one cluster

        for obs in observations:
            ox, oy = obs["world_x"], obs["world_y"]
            placed = False

            for cluster in clusters:
                # Cluster centre = running mean
                cx = sum(o["world_x"] for o in cluster) / len(cluster)
                cy = sum(o["world_y"] for o in cluster) / len(cluster)
                if math.sqrt((ox - cx) ** 2 + (oy - cy) ** 2) <= cluster_radius:
                    cluster.append(obs)
                    placed = True
                    break

            if not placed:
                clusters.append([obs])

        # The cluster with the most observations wins
        best_cluster = max(clusters, key=len)

        best_x = sum(o["world_x"] for o in best_cluster) / len(best_cluster)
        best_y = sum(o["world_y"] for o in best_cluster) / len(best_cluster)
        best_z = sum(o["world_z"] for o in best_cluster) / len(best_cluster)

        rospy.loginfo(
            f"   📊 Clusters: {[len(c) for c in clusters]} → "
            f"dominant cluster size={len(best_cluster)}, "
            f"avg=({best_x:.2f}, {best_y:.2f}, {best_z:.2f})"
        )

        return best_x, best_y, best_z, len(best_cluster)

    def save_object_detection(self, obj, world_coords, timestamp, current_pose, frame_id):
        """
        Stores every raw observation and keeps the canonical world position as
        the mean of the largest spatial cluster across all observations.

        First detection  → create a new entry with one observation.
        Subsequent detections → append observation, re-cluster, update position.
        """
        obj_class = obj["class"]

        new_obs = {
            "world_x": world_coords[0],
            "world_y": world_coords[1],
            "world_z": world_coords[2],
            "timestamp": timestamp,
        }

        # ── existing object: append + re-cluster ──────────────────────────────
        for existing in self.detected_objects.values():
            if existing["object_class"] == obj_class:

                existing.setdefault("all_observations", []).append(new_obs)
                existing["observation_count"] += 1

                result = self.cluster_observations(existing["all_observations"])
                if result:
                    best_x, best_y, best_z, cluster_size = result
                    existing["world_x"] = best_x
                    existing["world_y"] = best_y
                    existing["world_z"] = best_z
                    existing["dominant_cluster_size"] = cluster_size

                # Update other per-frame fields
                existing["pixel_x"]   = obj["center_x"]
                existing["pixel_y"]   = obj["center_y"]
                existing["bbox"]      = obj["bbox"]
                existing["timestamp"] = timestamp
                existing["confidence"]= obj["confidence"]
                existing["frame_id"]  = frame_id

                rospy.loginfo(
                    f"🔄 Updated {obj_class} "
                    f"(obs #{existing['observation_count']}) → "
                    f"clustered pos ({existing['world_x']:.2f}, "
                    f"{existing['world_y']:.2f})"
                )

                self.write_all_objects_to_file()
                return

        # ── brand-new object ───────────────────────────────────────────────────
        unique_id = f"obj_{self.object_id:06d}"
        self.object_id += 1

        self.detected_objects[unique_id] = {
            "id":               unique_id,
            "timestamp":        timestamp,
            "object_class":     obj_class,
            "confidence":       obj["confidence"],
            "detection_method": obj["detection_method"],
            "world_x":          world_coords[0],
            "world_y":          world_coords[1],
            "world_z":          world_coords[2],
            "pixel_x":          obj["center_x"],
            "pixel_y":          obj["center_y"],
            "bbox":             obj["bbox"],
            "robot_x":          current_pose.position.x,
            "robot_y":          current_pose.position.y,
            "robot_z":          current_pose.position.z,
            "frame_id":         frame_id,
            "observation_count":      1,
            "dominant_cluster_size":  1,
            "all_observations": [new_obs],
        }

        rospy.loginfo(
            f"🎯 New {obj_class} → "
            f"({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f})"
        )

        self.write_all_objects_to_file()

        
    def write_all_objects_to_file(self):
        """Overwrite file with current object dictionary"""
        try:
            with open(self.objects_file, "w") as f:
                for obj in self.detected_objects.values():
                    f.write(json.dumps(obj) + "\n")
        except Exception as e:
            rospy.logerr(f"Error writing objects file: {e}")
    

    def load_existing_objects(self):
        """Load previously detected objects from file"""
        self.detected_objects = {}
        self.object_id = 0

        if not os.path.exists(self.objects_file):
            return

        with open(self.objects_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.detected_objects[obj["id"]] = obj

                try:
                    num = int(obj["id"].split("_")[1])
                    self.object_id = max(self.object_id, num + 1)
                except Exception:
                    pass

        rospy.loginfo(f"📂 Loaded {len(self.detected_objects)} objects from file")



    def save_detection_visualization(self, cv_image, detected_objects, frame_id):
        """Save image with detection boxes for debugging"""
        annotated = cv_image.copy()
        
        for obj in detected_objects:
            x, y = obj['center_x'], obj['center_y']
            cv2.line(annotated, (x-20, y), (x+20, y), (0, 255, 0), 2)
            cv2.line(annotated, (x, y-20), (x, y+20), (0, 255, 0), 2)
            cv2.putText(annotated, obj['class'], (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        viz_path = os.path.join(self.save_dir, f"detection_{frame_id:05d}.png")
        cv2.imwrite(viz_path, annotated)
        rospy.loginfo(f"Saved detection visualization: {viz_path}")
    


    def get_objects_by_class(self, object_class):
        """Retrieve all objects of a specific class"""
        return [obj for obj in self.detected_objects.values() 
                if obj["object_class"] == object_class]
    

    def find_objects_near_location(self, x, y, radius=2.0, object_class=None):
        """Find objects near a specific location"""
        objects = []
        for obj in self.detected_objects.values():
            distance = math.sqrt((obj["world_x"] - x)**2 + (obj["world_y"] - y)**2)
            if distance <= radius:
                if object_class is None or obj["object_class"] == object_class:
                    objects.append(obj)
        return objects