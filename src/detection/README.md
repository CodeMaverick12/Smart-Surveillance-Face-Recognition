# Face Detector for Smart Surveillance System

This module provides **face detection and embedding extraction** for the Smart Surveillance System. It combines:

- **OpenCV DNN (SSD)** for fast and stable face detection.
- **ONNX ArcFace model** for extracting 512-dimensional embeddings.

---

## Features

- Detects faces in images or video frames.
- Extracts normalized 512-dimensional face embeddings.
- Returns bounding boxes along with placeholder age and gender.
- Fast detection on CPU with high-quality embeddings suitable for recognition.
- Designed to integrate seamlessly with tracking and identity management.

---

## Key Class

### `FaceDetector`
Handles both face detection and embedding extraction.

**Initialization**
```python
fd = FaceDetector()
