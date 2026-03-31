import requests
import cv2
import numpy as np

class MoonVLMDetector:
    def __init__(self, server_url="http://0.0.0.0:8000"):
        self.url_query = f"{server_url}/query"
        self.url_detect = f"{server_url}/detect"

    def detect(self, image, model_names):
        """
        Detects and classifies objects in the image using MoonVLM.
        Returns a list of dictionaries:
        [
            {
                "class": str,
                "bbox": [x1, y1, x2, y2],
                "center_x": int,
                "center_y": int,
                "confidence": float,
                "detection_method": str
            }
        ]
        """

        allowed_list_str = ", ".join(model_names)
        height, width = image.shape[:2]
        _, buffer = cv2.imencode(".jpg", image)
        files = {"image": ("image.jpg", buffer.tobytes(), "image/jpeg")}

       # ===== STEP 1: Query which allowed labels are present =====
        present_objects_prompt = (
            f"""Look at the image carefully and list ALL objects you can see from this allowed list: {allowed_list_str}.

        STRICT RULES:
        - Output ONLY labels from the allowed list, separated by commas.
        - Include every object from the list that is visible, even partially.
        - Do NOT label walls, floors, ceilings, brick textures, or any background surfaces.
        - Do NOT label textures or construction materials (e.g., concrete, brick, tiles) as objects.
        - Do NOT guess or invent labels not in the allowed list.
        - If nothing from the list is visible, output: NOTHING

        Output:"""
        )
        try:
            query_resp = requests.post(self.url_query, data={"question": present_objects_prompt}, files=files, timeout=30)
            query_resp.raise_for_status()
            found_labels_raw = query_resp.json().get("answer", "").lower()
        except requests.RequestException as e:
            print(f"[MoonVLM] Initial /query failed: {e}")
            return []

        # Filter our model_names to only those the VLM confirmed seeing
        active_targets = [name for name in model_names if name.lower() in found_labels_raw]
        
        # ===== STEP 2: Detect only the confirmed objects =====
        final_results = []
        for obj_name in active_targets:
            try:
                detect_resp = requests.post(self.url_detect, data={"query": obj_name}, files=files, timeout=30)
                detect_resp.raise_for_status()
                detections = detect_resp.json().get("objects", [])
            except requests.RequestException:
                continue
            

            for obj in detections:
                x1, y1, x2, y2 = obj["bbox"]
                # Convert to pixel coordinates
                px1, py1 = int(x1 * width), int(y1 * height)
                px2, py2 = int(x2 * width), int(y2 * height)

                if 0 <= x1 <= 1 and 0 <= y1 <= 1:
                    px1 = int(x1 * width)
                    py1 = int(y1 * height)
                    px2 = int(x2 * width)
                    py2 = int(y2 * height)
                else:
                    px1, py1, px2, py2 = map(int, obj["bbox"]) 

                area = (px2 - px1) * (py2 - py1)

                if area > 0.75 * width * height:
                  continue 

            
                if not self._verify_detection(image, [px1, py1, px2, py2], obj_name):
                    print(f"[MoonVLM] Rejected false positive: '{obj_name}' at [{px1},{py1},{px2},{py2}]")
                    continue

                final_results.append({
                    "class": obj_name,
                    "bbox": [px1, py1, px2, py2],
                    "center_x": (px1 + px2) // 2,
                    "center_y": (py1 + py2) // 2,
                    "confidence": float(obj.get("score", 0.7)),
                    "detection_method": "query_first_detect"
                })

        return final_results

    def _verify_detection(self, image, bbox, obj_name):
        """Crop the detected region and verify the label is correct."""
        px1, py1, px2, py2 = bbox
        
        # Add padding around crop
        h, w = image.shape[:2]
        pad = 10
        crop = image[
            max(0, py1 - pad):min(h, py2 + pad),
            max(0, px1 - pad):min(w, px2 + pad)
        ]
        
        if crop.size == 0:
            return False

        _, buffer = cv2.imencode(".jpg", crop)
        files = {"image": ("crop.jpg", buffer.tobytes(), "image/jpeg")}

        verify_prompt = (
            f"""Does this image clearly show a "{obj_name}"?
    Answer with ONLY "yes" or "no".
    Do NOT say yes if it is a similar but different object (e.g. fire hydrant ≠ fire extinguisher).
    Answer:"""
        )

        try:
            resp = requests.post(self.url_query, data={"question": verify_prompt}, files=files, timeout=30)
            resp.raise_for_status()
            answer = resp.json().get("answer", "").lower().strip()
            return answer.startswith("yes")
        except requests.RequestException:
            return True  # If verification fails, keep the detection

    # def detect(self, image, model_names):
    #     """
    #     Detects and classifies objects in the image using MoonVLM.
    #     Returns a list of dictionaries:
    #     [
    #         {
    #             "class": str,
    #             "bbox": [x1, y1, x2, y2],
    #             "center_x": int,
    #             "center_y": int,
    #             "confidence": float,
    #             "detection_method": str
    #         }
    #     ]
    #     """
    #     allowed_list_str = ", ".join(model_names)
    #     height, width = image.shape[:2]
    #     _, buffer = cv2.imencode(".jpg", image)
    #     files = {"image": ("image.jpg", buffer.tobytes(), "image/jpeg")}

    #     # ===== STEP 1: Query which allowed labels are present =====
    #     objects = []

    #     for obj_name in model_names:
    #         detect_question = f"{obj_name}"
    #         detect_data = {"query": detect_question}

    #         try:
    #             detect_resp = requests.post(self.url_detect, data=detect_data, files=files, timeout=30)
    #             detect_resp.raise_for_status()
    #         except requests.RequestException as e:
    #             print(f"[MoonVLM] /detect failed for '{obj_name}': {e}")
    #             continue

    #         detect_json = detect_resp.json()
    #         detections = detect_json.get("objects", [])

    #         for obj in detections:
    #             if "bbox" in obj:
    #                 x1, y1, x2, y2 = obj["bbox"]
    #                 objects.append({
    #                     "object_name": obj_name,
    #                     "bbox": [x1, y1, x2, y2],
    #                     "other_info": {k: v for k, v in obj.items() if k != "bbox"}
    #                 })
    #             else:
    #                 print(f"[MoonVLM] Warning: 'bbox' not found for object '{obj_name}' in response")

    #     # ===== STEP 2: Classify each detected object =====
    #     final_results = []

    #     for obj in objects:
    #         x1, y1, x2, y2 = obj["bbox"]

    #         # Convert bbox from relative to pixel coordinates if needed
    #         if 0 <= x1 <= 1 and 0 <= y1 <= 1:
    #             px1 = int(x1 * width)
    #             py1 = int(y1 * height)
    #             px2 = int(x2 * width)
    #             py2 = int(y2 * height)
    #         else:
    #             px1, py1, px2, py2 = map(int, obj["bbox"])

    #         # Skip very large objects (likely walls or floors)
    #         area = (px2 - px1) * (py2 - py1)
    #         if area > 0.75 * width * height:
    #             continue

    #         # Safety clamp
    #         px1, py1 = max(0, px1), max(0, py1)
    #         px2, py2 = min(width, px2), min(height, py2)

    #         crop = image[py1:py2, px1:px2]
    #         if crop.size == 0:
    #             continue

    #         # Encode crop for /query
    #         _, crop_buf = cv2.imencode(".jpg", crop)
    #         crop_files = {"image": ("crop.jpg", crop_buf.tobytes(), "image/jpeg")}

    #         classify_prompt = (
    #             f"You are classifying ONE cropped object image.\n\n"
    #             f"Allowed labels (ONLY these):\n{allowed_list_str}\n\n"
    #             "Task:\nReturn ONE label from the allowed list, OR return NOTHING.\n\n"
    #             "STRICT RULES:\n"
    #             "- Output EXACTLY ONE label from the allowed list, OR NOTHING.\n"
    #             "- Skip walls, bricks, floors, ceilings, or background surfaces.\n"
    #             "- DO NOT guess or invent labels. If it contains bricks, just skip that object\n"
    #             "Output:\n"
    #         )

    #         try:
    #             query_resp = requests.post(self.url_query, data={"question": classify_prompt}, files=crop_files, timeout=30)
    #             query_resp.raise_for_status()
    #         except requests.RequestException as e:
    #             print(f"[MoonVLM] /query failed: {e}")
    #             continue

    #         label = query_resp.json().get("answer", "").strip()

    #         # Map to allowed labels
    #         best_label = None
    #         for allowed in model_names:
    #             if allowed.lower() in label.lower():
    #                 best_label = allowed
    #                 break
    #         if best_label is None:
    #             continue

    #         # Compute center
    #         cx, cy = (px1 + px2) // 2, (py1 + py2) // 2

    #         final_results.append({
    #             "class": best_label,
    #             "bbox": [px1, py1, px2, py2],
    #             "center_x": cx,
    #             "center_y": cy,
    #             "confidence": float(obj.get("score", 0.5)),
    #             "detection_method": "detect+query"
    #         })

    #     best_per_class = {}

    #     for det in final_results:
    #         cls = det["class"]

    #         if cls not in best_per_class or det["confidence"] > best_per_class[cls]["confidence"]:
    #             best_per_class[cls] = det

    #     def iou(boxA, boxB):
    #         xA = max(boxA[0], boxB[0])
    #         yA = max(boxA[1], boxB[1])
    #         xB = min(boxA[2], boxB[2])
    #         yB = min(boxA[3], boxB[3])
            
    #         interW = max(0, xB - xA)
    #         interH = max(0, yB - yA)
    #         interArea = interW * interH

    #         boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    #         boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    #         union = boxAArea + boxBArea - interArea
    #         return interArea / union if union > 0 else 0

    #     # Sort by confidence descending
    #     filtered = []
    #     for det in sorted(final_results, key=lambda x: x["confidence"], reverse=True):
    #         keep = True
    #         for f in filtered:
    #             if iou(det["bbox"], f["bbox"]) > 0.5:   # 0.5 = overlap threshold
    #                 keep = False
    #                 break
    #         if keep:
    #             filtered.append(det)

    #     return filtered



