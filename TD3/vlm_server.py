# # vlm_server.py
# import time
# from fastapi import FastAPI, UploadFile, Form
# from PIL import Image
# import torch
# import io
# from transformers import AutoModelForCausalLM
# from typing import Optional, Union

# app = FastAPI()

# MODEL_ID = "vikhyatk/moondream2"
# REVISION = "2025-01-09"

# print("[*] Loading VLM model...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     revision=REVISION,
#     trust_remote_code=True,
#     device_map={"": "cuda" if torch.cuda.is_available() else "cpu"}
# )
# model.eval()
# print("[*] Model loaded.")

# @app.post("/query")
# async def query_vlm(image: UploadFile, question: str = Form(...)):
#     image_data = await image.read()
#     pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

#     raw = model.query(pil_image, question)

#     # Normalize to a plain string. Some Moondream builds return {"answer": "..."}.
#     if isinstance(raw, dict):
#         raw = raw.get("answer", raw)
#     if not isinstance(raw, str):
#         raw = str(raw)

#     return {"answer": raw}


# vlm_server.py
import io
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

"""
Moondream VLM server with 3 endpoints:

POST /query   -> {"answer": "..."}
POST /point   -> {"points": [{"x": int, "y": int, "score": float}, ...]}
POST /detect  -> {"objects": [{"bbox":[x1,y1,x2,y2], "label": str, "score": float}, ...]}

- Uses model.query(image, question) if available (recommended).
- Gracefully falls back to encode_image + query(enc, question) on older revisions.
- For /point and /detect, uses model.point / model.detect when present.
  Otherwise, it falls back to /query with a strict JSON instruction and parses.
"""

MODEL_ID = os.environ.get("MOONDREAM_MODEL", "vikhyatk/moondream2")
REVISION = os.environ.get("MOONDREAM_REVISION", None)  # pin a revision if you want

app = FastAPI(title="Moondream VLM Server", version="1.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model() -> AutoModelForCausalLM:
    print(f"[*] Loading VLM model: {MODEL_ID} on {DEVICE} ...")
    kwargs = dict(trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.to(DEVICE)  # move entire model to GPU or CPU
    model.eval()
    print("[*] Model loaded.")
    return model

MODEL = _load_model()

def _ensure_pil(image_file: UploadFile) -> Image.Image:
    data = image_file.file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return pil


def _query_answer(pil: Image.Image, question: str) -> str:
    try:
        if hasattr(MODEL, "query"):
            # pass PIL directly
            out = MODEL.query(pil, question)
            if isinstance(out, dict) and "answer" in out:
                return str(out["answer"])
            return str(out)

        elif hasattr(MODEL, "encode_image"):
            enc = MODEL.encode_image(pil)  # EncodedImage object
            out = MODEL.query(enc, question)  # do NOT call .to() here
            if isinstance(out, dict) and "answer" in out:
                return str(out["answer"])
            return str(out)

        else:
            return "The model does not support `query()`."
    except Exception as e:
        print("[ERROR] Query failed:", e)
        raise e




def _point_objects(pil: Image.Image, query: str) -> Dict[str, Any]:
    """
    Preferred: model.point(pil, query) -> {"points":[{"x":...,"y":...,"score":...}, ...]}
    Fallback:  query() with strict JSON instruction and parse minimal structure.
    """
    if hasattr(MODEL, "point"):
        out = MODEL.point(pil, query)
        # Expecting a dict; pass through
        return out if isinstance(out, dict) else {"points": []}

    # Fallback: force a strict JSON answer
    prompt = (
        f"Given this image, return pixel locations for '{query}'. "
        "Respond ONLY with JSON: {\"points\":[{\"x\":int,\"y\":int,\"score\":0.xx}, ...]}"
    )
    ans = _query_answer(pil, prompt)
    try:
        import json, re
        blob = ans if ans.strip().startswith("{") else re.search(r"\{.*\}", ans, re.S).group(0)
        j = json.loads(blob)
        return j if isinstance(j, dict) and "points" in j else {"points": []}
    except Exception:
        return {"points": []}


def _detect_objects(pil: Image.Image, query: str) -> Dict[str, Any]:
    """
    Preferred: model.detect(pil, query) -> {"objects":[{"bbox":[x1,y1,x2,y2], "label":str, "score":float}, ...]}
    Fallback:  query() with strict JSON instruction and parse minimal structure.
    """
    if hasattr(MODEL, "detect"):
        out = MODEL.detect(pil, query)
        return out if isinstance(out, dict) else {"objects": []}

    # Fallback via strict JSON
    prompt = (
        f"Detect '{query}' in this image. "
        "Respond ONLY with JSON: {\"objects\":[{\"bbox\":[x1,y1,x2,y2],\"label\":\"...\",\"score\":0.xx}, ...]}"
    )
    ans = _query_answer(pil, prompt)
    try:
        import json, re
        blob = ans if ans.strip().startswith("{") else re.search(r"\{.*\}", ans, re.S).group(0)
        j = json.loads(blob)
        return j if isinstance(j, dict) and "objects" in j else {"objects": []}
    except Exception:
        return {"objects": []}


@app.post("/query")
async def query_vlm(image: UploadFile, question: str = Form(...)):
    pil = _ensure_pil(image)
    answer = _query_answer(pil, question)
    return {"answer": answer}


@app.post("/point")
async def point(image: UploadFile, query: str = Form(...)):
    pil = _ensure_pil(image)
    out = _point_objects(pil, query)
    return JSONResponse(out)

@app.post("/detect")
async def detect(image: UploadFile, query: str = Form(...)):
    import io
    from PIL import Image

    data = await image.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = pil_img.size

    detections = []

    if not hasattr(MODEL, "detect"):
        return {"objects": []}

    raw = MODEL.detect(pil_img, query)
    objects = raw.get("objects", [])

    for obj in objects:
        if not all(k in obj for k in ("x_min", "y_min", "x_max", "y_max")):
            continue

        x1 = int(obj["x_min"] * width)
        y1 = int(obj["y_min"] * height)
        x2 = int(obj["x_max"] * width)
        y2 = int(obj["y_max"] * height)

        # ---------- WALL FILTERS ----------
        area = (x2 - x1) * (y2 - y1)
        img_area = width * height

        # 1️⃣ Large area → wall
        if area > 0.35 * img_area:
            continue

        # 2️⃣ Edge-touching → background
        if x1 < 10 or y1 < 10 or x2 > width - 10 or y2 > height - 10:
            continue

        # 3️⃣ Extreme aspect ratio → wall
        w = x2 - x1
        h = y2 - y1
        if w / max(h, 1) > 6 or h / max(w, 1) > 6:
            continue
        # ---------------------------------

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "center_x": cx,
            "center_y": cy
        })

    return {"objects": detections}
