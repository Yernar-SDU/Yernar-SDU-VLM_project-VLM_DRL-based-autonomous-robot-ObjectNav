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
import csv

class GPT_VLMDetector:
    """
    Handles all VLM-based object detection and coordinate transformation
    """
    def __init__(self):
        # VLM setup
        self.client = None
        # OpenAI API key - SET YOUR KEY HERE
        api_key = "YOUR OWN API KEY"
        self.client = OpenAI(api_key=api_key)
    
    def detect_with_vlm(self, image, model_names):
        """Object detection using Vision Language Model"""
        try:
            height, width = image.shape[:2]
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()
            
            model_list_formatted = " - " + "\n - ".join(model_names) 
            
            prompt = f"""
                You are a precise object localization system. Analyze this {width}x{height} image.

                TASK: Identify standalone, movable objects from the list:{model_list_formatted}

                INSTRUCTIONS:
                • You may only report objects using exact names from the list above.
                • If an object is present but not listed, ignore it.
                • If unsure about mapping, skip instead of guessing.

                For EACH detected object, return:
                1. Exact model name (must match list spelling character-for-character)
                2. Pixel X center of the object (0 = left, {width} = right)
                3. Pixel Y center of the object (0 = top, {height} = bottom)

                FORMAT (one per line): object_name:[x,y]

                Now analyze the image:
            """
        
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ]
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
                            'confidence': 0.8,
                            'center_x': x,
                            'center_y': y,
                            'bbox': [x-25, y-25, x+25, y+25],
                            'detection_method': 'GPT_vlm'
                        }
                        objects.append(obj)
                        rospy.loginfo(f"VLM detected {object_name} at pixel: ({x}, {y})")
            except Exception as e:
                rospy.logwarn(f"Failed to parse line: {line}, error: {e}")
                continue
        
        return objects
  