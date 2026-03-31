import csv
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os

def extract_model_positions(world_path):
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

world_file = "/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch/TD3_signs2.world"
output_csv = "_models.csv"

data =  extract_model_positions(world_file)
with open(output_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Model Name", "X", "Y", "Z"])
            writer.writerows(data)
