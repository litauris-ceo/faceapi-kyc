import os
import tempfile
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from insightface.app import FaceAnalysis

face_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    model_name = os.environ.get("INSIGHTFACE_MODEL", "buffalo_sc")
    face_app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"InsightFace models loaded ({model_name})")
    yield


app = FastAPI(lifespan=lifespan)


def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def extract_face_from_document(img: np.ndarray):
    """Detect face in a document photo. If the face is small relative to
    the image (typical for ID/passport), crop and upscale the face region
    then re-analyse for a better quality embedding."""

    faces = face_app.get(img)
    if not faces:
        for scale in [1.5, 2.0]:
            h, w = img.shape[:2]
            resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            faces = face_app.get(resized)
            if faces:
                break
    if not faces:
        return None

    face = max(faces, key=lambda f: f.det_score)

    bbox = face.bbox.astype(int)
    img_h, img_w = img.shape[:2]
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    face_area_ratio = (face_w * face_h) / (img_w * img_h)

    if face_area_ratio < 0.15:
        pad_w = int(face_w * 0.5)
        pad_h = int(face_h * 0.5)
        x1 = max(0, bbox[0] - pad_w)
        y1 = max(0, bbox[1] - pad_h)
        x2 = min(img_w, bbox[2] + pad_w)
        y2 = min(img_h, bbox[3] + pad_h)

        crop = img[y1:y2, x1:x2]

        target = 640
        crop_h, crop_w = crop.shape[:2]
        scale = target / max(crop_h, crop_w)
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        crop_faces = face_app.get(crop)
        if crop_faces:
            return max(crop_faces, key=lambda f: f.det_score)

    return face


def extract_face_from_selfie(img: np.ndarray):
    faces = face_app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.environ.get("INSIGHTFACE_MODEL", "buffalo_sc")}


@app.post("/compare")
async def compare(
    document: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    doc_bytes = await document.read()
    selfie_bytes = await selfie.read()

    try:
        doc_img = read_image(doc_bytes)
    except ValueError:
        raise HTTPException(400, "Could not decode document image")

    try:
        selfie_img = read_image(selfie_bytes)
    except ValueError:
        raise HTTPException(400, "Could not decode selfie image")

    doc_face = extract_face_from_document(doc_img)
    if doc_face is None:
        raise HTTPException(422, "No face detected in document photo")

    selfie_face = extract_face_from_selfie(selfie_img)
    if selfie_face is None:
        raise HTTPException(422, "No face detected in selfie")

    similarity = cosine_similarity(doc_face.embedding, selfie_face.embedding)

    threshold = 0.4
    match = similarity >= threshold

    confidence = max(0.0, min(1.0, (similarity - threshold) / (1 - threshold))) if match else max(0.0, similarity / threshold)

    return {
        "match": match,
        "similarity": round(similarity, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "document": {
            "face_detected": True,
            "detection_score": round(float(doc_face.det_score), 4),
            "bbox": doc_face.bbox.astype(int).tolist(),
        },
        "selfie": {
            "face_detected": True,
            "detection_score": round(float(selfie_face.det_score), 4),
            "bbox": selfie_face.bbox.astype(int).tolist(),
        },
    }


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KYC Face Compare</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 16px rgba(0,0,0,.1); padding: 32px; max-width: 540px; width: 100%; }
  h1 { font-size: 22px; margin-bottom: 4px; text-align: center; }
  .subtitle { font-size: 13px; color: #888; text-align: center; margin-bottom: 24px; }
  .uploads { display: flex; gap: 16px; margin-bottom: 24px; }
  .upload-box { flex: 1; border: 2px dashed #ccc; border-radius: 8px; padding: 16px; text-align: center; cursor: pointer; position: relative; transition: border-color .2s; min-height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .upload-box:hover { border-color: #4a90d9; }
  .upload-box.has-file { border-color: #4a90d9; border-style: solid; }
  .upload-box input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-box .label { font-size: 14px; color: #666; margin-top: 8px; }
  .upload-box .sublabel { font-size: 11px; color: #aaa; margin-top: 2px; }
  .upload-box .name { font-size: 12px; color: #333; margin-top: 4px; word-break: break-all; }
  .upload-box img.preview { max-width: 100%; max-height: 120px; border-radius: 4px; object-fit: contain; }
  button { width: 100%; padding: 12px; background: #4a90d9; color: #fff; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; transition: background .2s; }
  button:hover { background: #357abd; }
  button:disabled { background: #b0c4de; cursor: not-allowed; }
  .result { margin-top: 24px; padding: 20px; border-radius: 8px; text-align: center; display: none; }
  .result.match { background: #e6f9e6; border: 1px solid #4caf50; }
  .result.no-match { background: #fdecea; border: 1px solid #f44336; }
  .result.error { background: #fff3e0; border: 1px solid #ff9800; }
  .result .verdict { font-size: 20px; font-weight: 700; }
  .result .score { font-size: 14px; color: #555; margin-top: 8px; }
  .spinner { display: none; margin: 24px auto 0; width: 32px; height: 32px; border: 3px solid #e0e0e0; border-top-color: #4a90d9; border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>KYC Face Comparison</h1>
  <div class="subtitle">InsightFace ArcFace model</div>
  <form id="form">
    <div class="uploads">
      <div class="upload-box" id="doc-box">
        <input type="file" accept="image/*" id="doc-input">
        <div class="label">Document Photo</div>
        <div class="sublabel">Passport / Driving License</div>
        <div class="name" id="doc-name"></div>
      </div>
      <div class="upload-box" id="selfie-box">
        <input type="file" accept="image/*" id="selfie-input">
        <div class="label">Selfie</div>
        <div class="sublabel">Clear face photo</div>
        <div class="name" id="selfie-name"></div>
      </div>
    </div>
    <button type="submit" id="btn" disabled>Compare Faces</button>
  </form>
  <div class="spinner" id="spinner"></div>
  <div class="result" id="result">
    <div class="verdict" id="verdict"></div>
    <div class="score" id="score"></div>
  </div>
</div>
<script>
  const docInput = document.getElementById('doc-input');
  const selfieInput = document.getElementById('selfie-input');
  const btn = document.getElementById('btn');

  function setupPreview(input, boxId, nameId) {
    input.addEventListener('change', () => {
      const box = document.getElementById(boxId);
      const nameEl = document.getElementById(nameId);
      box.querySelectorAll('img.preview').forEach(e => e.remove());
      if (input.files[0]) {
        box.classList.add('has-file');
        nameEl.textContent = input.files[0].name;
        const img = document.createElement('img');
        img.className = 'preview';
        img.src = URL.createObjectURL(input.files[0]);
        box.insertBefore(img, box.firstChild);
      } else {
        box.classList.remove('has-file');
        nameEl.textContent = '';
      }
      btn.disabled = !(docInput.files[0] && selfieInput.files[0]);
    });
  }
  setupPreview(docInput, 'doc-box', 'doc-name');
  setupPreview(selfieInput, 'selfie-box', 'selfie-name');

  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const result = document.getElementById('result');
    const spinner = document.getElementById('spinner');
    result.style.display = 'none';
    spinner.style.display = 'block';
    btn.disabled = true;

    const fd = new FormData();
    fd.append('document', docInput.files[0]);
    fd.append('selfie', selfieInput.files[0]);

    try {
      const res = await fetch('/compare', { method: 'POST', body: fd });
      const data = await res.json();
      spinner.style.display = 'none';
      result.style.display = 'block';

      if (data.detail) {
        result.className = 'result error';
        document.getElementById('verdict').textContent = 'Error';
        document.getElementById('score').textContent = data.detail;
      } else if (data.match) {
        result.className = 'result match';
        document.getElementById('verdict').textContent = 'MATCH';
        document.getElementById('score').textContent =
          'Similarity: ' + (data.similarity * 100).toFixed(1) + '% | Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
      } else {
        result.className = 'result no-match';
        document.getElementById('verdict').textContent = 'NO MATCH';
        document.getElementById('score').textContent =
          'Similarity: ' + (data.similarity * 100).toFixed(1) + '% | Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
      }
    } catch (err) {
      spinner.style.display = 'none';
      result.style.display = 'block';
      result.className = 'result error';
      document.getElementById('verdict').textContent = 'Error';
      document.getElementById('score').textContent = 'Request failed';
    }
    btn.disabled = !(docInput.files[0] && selfieInput.files[0]);
  });
</script>
</body>
</html>"""
