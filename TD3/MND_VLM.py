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
            f"""Return EXACTLY ONE label from the allowed list {allowed_list_str} that correctly identifies the object in the image.
                If the image does not contain any valid object from the allowed list, return NOTHING.

                STRICT RULES:

                Output exactly ONE label from the allowed list, or NOTHING.

                Do NOT label walls, floors, ceilings, brick textures, or any background surfaces.

                Do NOT label textures or construction materials (e.g., concrete, brick, tiles) as objects.

                Do NOT guess or invent labels.

                Output:
                """
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

                final_results.append({
                    "class": obj_name,
                    "bbox": [px1, py1, px2, py2],
                    "center_x": (px1 + px2) // 2,
                    "center_y": (py1 + py2) // 2,
                    "confidence": float(obj.get("score", 0.7)),
                    "detection_method": "query_first_detect"
                })



        return final_results

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
    #             "- Skip walls, floors, ceilings, or background surfaces.\n"
    #             "- DO NOT guess or invent labels.\n"
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

    #     return final_results
