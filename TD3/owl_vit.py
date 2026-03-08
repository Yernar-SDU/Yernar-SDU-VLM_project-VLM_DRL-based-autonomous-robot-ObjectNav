import requests
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
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


# Load model + processor
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model     = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load image

image_path = "/home/ai-lab/Downloads/умные кошки1.png"
image = Image.open(image_path).convert("RGB")

# Define text queries (for anchor of labels)
text_queries = ["a photo of a cat", "a photo of a dog"]

# Preprocess
inputs  = processor(text=[text_queries], images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Prepare size
target_sizes = torch.tensor([(image.height, image.width)])

# Post-process (no text_labels arg)
results = processor.post_process_object_detection(
    outputs=outputs,
    threshold=0.1,
    target_sizes=target_sizes
)

# Extract and compute centers
result = results[0]  # first image in batch
boxes  = result["boxes"]
scores = result["scores"]
labels = result["labels"]

for box, score, label_idx in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box.tolist()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    label_text = text_queries[label_idx] if label_idx < len(text_queries) else f"label_{label_idx}"
    print(f"Detected {label_text} with confidence {score.item():.3f}")
    print(f"Bounding box: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")
    print(f"Center pixel: (x={cx:.2f}, y={cy:.2f})")

# Optional: draw on image
draw = ImageDraw.Draw(image)
for box, label_idx in zip(boxes, labels):
    xmin, ymin, xmax, ymax = box.tolist()
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill="blue")
image.show()
