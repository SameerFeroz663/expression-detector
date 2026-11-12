"""
FastAPI server for facial expression recognition.

Pipeline for each request:
  image bytes -> PIL RGB image -> Haar cascade face detection
  -> square face crops -> training-identical preprocessing
  -> ResNet18 (one batch for all faces) -> softmax -> JSON

Run with:  uvicorn main:app --port 8000
"""

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "emotion_model.pt"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
DETECT_MAX_WIDTH = 640  # wider images are downscaled before face detection
CROP_PADDING = 0.10  # ~10% extra margin on each side of the detected face
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Everything below is filled in once at startup (see lifespan) and reused.
state: dict = {}

# OpenCV's cascade and PyTorch inference are CPU-bound and run in FastAPI's
# thread pool; a lock keeps concurrent requests from sharing them at once.
inference_lock = threading.Lock()


def load_model():
    """Rebuild the network exactly as in training, then load the weights."""
    # weights_only=True is the safe way to load: it only unpickles tensors and
    # plain Python types (our dict of weights + list of class names).
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = list(checkpoint["classes"])  # e.g. ["angry", "disgust", ...]

    # Same architecture as training: a standard ResNet18 whose last layer
    # (fc) is swapped for one with len(classes) outputs instead of 1000.
    # weights=None because we're about to load our own trained weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state"])

    # eval() switches BatchNorm/Dropout to inference behaviour. Forgetting it
    # gives wrong (and batch-dependent) predictions.
    model.eval()
    return model, classes


def build_preprocess():
    """Preprocessing that MUST match training exactly.

    1. Grayscale(3): FER-2013 is grayscale, so the model was trained on gray
       faces copied into 3 channels (ResNet expects 3-channel input).
    2. ResNet18_Weights.DEFAULT.transforms(): resize shortest side to 256,
       center-crop 224x224, scale to [0, 1], normalize with ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        models.ResNet18_Weights.DEFAULT.transforms(),
    ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts, not on every request.
    model, classes = load_model()
    state["model"] = model
    state["classes"] = classes
    state["preprocess"] = build_preprocess()
    state["face_detector"] = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if state["face_detector"].empty():
        raise RuntimeError("Could not load OpenCV Haar cascade")
    print(f"Model loaded with classes: {classes}")
    yield
    state.clear()


app = FastAPI(title="Facial Expression Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return face boxes (x, y, w, h) in the ORIGINAL image's pixel coordinates."""
    width, height = image.size

    # Haar cascades are slow on big images, so detect on a smaller copy.
    scale = 1.0
    small = image
    if width > DETECT_MAX_WIDTH:
        scale = DETECT_MAX_WIDTH / width
        small = image.resize((DETECT_MAX_WIDTH, max(1, round(height * scale))))

    # The cascade works on grayscale images.
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    found = state["face_detector"].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    # Undo the downscale so the boxes match the original image.
    boxes = []
    for x, y, w, h in found:
        x0 = max(0, round(x / scale))
        y0 = max(0, round(y / scale))
        x1 = min(width, round((x + w) / scale))
        y1 = min(height, round((y + h) / scale))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def square_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Cut a square around the face with ~10% padding, clamped to the image.

    Why square + padding? Haar boxes are tight and slightly off-center, while
    FER-2013 faces are square crops that include a bit of forehead/chin.
    Matching that framing helps the model. Near the image border the square
    is clamped, so it may end up a little narrower there.
    """
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * CROP_PADDING)
    cx, cy = x + w / 2, y + h / 2
    left = max(0, round(cx - side / 2))
    top = max(0, round(cy - side / 2))
    right = min(image.width, round(cx + side / 2))
    bottom = min(image.height, round(cy + side / 2))
    return image.crop((left, top, right, bottom))


def classify_faces(crops: list[Image.Image]) -> list[list[float]]:
    """Run all face crops through the model in a single batch."""
    preprocess = state["preprocess"]
    batch = torch.stack([preprocess(c) for c in crops])  # shape [N, 3, 224, 224]
    with torch.no_grad():  # no gradients needed for inference: faster, less memory
        logits = state["model"](batch)  # shape [N, num_classes]
        probs = torch.softmax(logits, dim=1)  # raw scores -> probabilities summing to 1
    return probs.tolist()


def read_image(file: UploadFile | None) -> Image.Image:
    """Validate the upload and decode it into an RGB PIL image (in memory only)."""
    if file is None or not file.filename:
        raise HTTPException(400, "No file uploaded. Send an image in the 'file' field.")

    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("image/") or content_type == "application/octet-stream"
    ):
        raise HTTPException(400, f"File must be an image (got '{content_type}').")

    # Read at most 1 byte past the limit so a huge upload isn't fully loaded.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB.")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        # Apply the EXIF rotation tag (phone photos) so pixel coordinates match
        # how browsers display the image.
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise HTTPException(400, "Could not read the image. Is it a valid image file?")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "classes": state["classes"]}


# A plain `def` (not `async def`) makes FastAPI run this in a worker thread,
# so the CPU-heavy work doesn't block the server's event loop.
@app.post("/predict")
def predict(file: UploadFile | None = File(None)):
    image = read_image(file)
    width, height = image.size
    classes = state["classes"]

    with inference_lock:
        boxes = detect_faces(image)
        if not boxes:
            return {"image_width": width, "image_height": height, "faces": []}
        crops = [square_crop(image, b) for b in boxes]
        all_probs = classify_faces(crops)

    faces = []
    for box, probs in zip(boxes, all_probs):
        ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
        top_label, top_prob = ranked[0]
        faces.append({
            "box": [int(v) for v in box],
            "emotion": top_label,
            "confidence": round(top_prob, 4),
            "sure": top_prob >= CONFIDENCE_THRESHOLD,
            "probabilities": [
                {"label": label, "probability": round(p, 4)} for label, p in ranked
            ],
        })

    return {"image_width": width, "image_height": height, "faces": faces}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
