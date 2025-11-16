
---

## **README for `face_detector.py`**

```markdown
# Face Detector for Smart Surveillance System

This module provides **face detection and embedding extraction** for the Smart Surveillance System. It uses:

- **OpenCV DNN (SSD)** for fast and stable face detection.
- **ONNX ArcFace model** for extracting 512-dimensional embeddings.

## Features

- Detects faces in an image or video frame.
- Extracts normalized 512-D face embeddings.
- Returns bounding boxes along with placeholder age and gender.
- Fast detection on CPU with high-quality embeddings for recognition.

## Key Class

### `FaceDetector`
- Initializes face detection and recognition models.
- Methods:
  - `get_face_embeddings(frame, confidence_threshold=0.5)`:
    - Detects faces.
    - Crops and resizes to 112x112.
    - Extracts embeddings.
    - Returns a list of dictionaries:

```python
{
    'bbox': (startX, startY, endX, endY),
    'embedding': embedding_vector,
    'age': '?',
    'gender': '?'
}
