import torch
import cv2
from PIL import Image
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OwlVLMDetector:
    """
    Simple wrapper around OWL-ViT for text-query object detection,
    returning objects with class name, bbox, center_x/y, confidence
    """
    def __init__(self, text_queries=None, threshold=0.2, model_name="google/owlvit-base-patch32"):
        self.text_queries = text_queries or ["a photo of a cat", "a photo of a dog"]
        self.threshold = threshold
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model     = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        # if you have GPU:
        if torch.cuda.is_available():
            self.model.cuda()

    def detect(self, cv_image_bgr):
        """
        cv_image_bgr: OpenCV BGR image (numpy array)
        Returns: list of dicts with keys: class, confidence, bbox, center_x, center_y
        """
        # convert to RGB PIL image
        cv_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_rgb).convert("RGB")

        inputs = self.processor(text=[self.text_queries], images=pil_img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k:v.cuda() for k,v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(pil_img.height, pil_img.width)])

        # Use this when grounded function is unavailable
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )

        detections = []
        result = results[0]
        boxes  = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]  # class indices

        for box, score, label_idx in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box.tolist()
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            class_text = self.text_queries[label_idx] if label_idx < len(self.text_queries) else f"label_{label_idx}"
            detections.append({
                "class": class_text,
                "confidence": float(score.item()),
                "bbox": [xmin, ymin, xmax, ymax],
                "center_x": int(round(cx)),
                "center_y": int(round(cy)),
                "detection_method": "owl_vlm"
            })


        return detections
