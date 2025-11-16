# detection/face_detector.py

import onnxruntime
import numpy as np
import cv2
import os

# Get the absolute path to the current file's directory (src/detection/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Models are in src/detection/models/
MODEL_DIR = os.path.join(current_dir, 'models')
DETECTION_PROTO = os.path.join(MODEL_DIR, 'deploy.prototxt')
DETECTION_MODEL = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
RECOGNITION_MODEL = os.path.join(MODEL_DIR, 'arcface_r50.onnx')
INPUT_SIZE = (112, 112)  # Required input size for ArcFace
class FaceDetector:
    """
    Handles Face Detection (CV2 DNN) and Embedding (ONNX ArcFace).
    """
    def __init__(self):
        print("Initializing Stable CV Face Detector and ONNX ArcFace Recognizer...")
        
        # 1. Initialize CV2 DNN Detector (Fast and stable on CPU)
        self.detector = cv2.dnn.readNetFromCaffe(DETECTION_PROTO, DETECTION_MODEL)
        
        # 2. Initialize ONNX Runtime Recognizer (High Accuracy)
        opts = onnxruntime.SessionOptions()
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL 
        self.rec_session = onnxruntime.InferenceSession(RECOGNITION_MODEL, opts, providers=['CPUExecutionProvider'])
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
    def get_face_embeddings(self, frame: np.ndarray, confidence_threshold: float = 0.5):
        """Processes a frame: Detects faces, aligns them (via center crop), and extracts embeddings."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces_data = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # --- CRITICAL COMPROMISE: Alignment via Simple Crop (Fastest Stable Way) ---
                face_crop = frame[startY:endY, startX:endX]
                if face_crop.size == 0: continue
                
                # Resize to the required 112x112 for the ArcFace model
                aligned_face = cv2.resize(face_crop, INPUT_SIZE)
                
                # Extract Embedding
                embedding = self._get_embedding(aligned_face)

                # Placeholder for Age/Gender (We'll add a separate lightweight model later)
                # For now, we'll mark these as None/Unknown to ensure the system is complete.
                
                faces_data.append({
                    'bbox': (startX, startY, endX, endY),
                    'embedding': embedding,
                    'age': '?',
                    'gender': '?'
                })
                
        return faces_data

    def _get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Generates the 512D embedding."""
        
        # Preprocessing: ArcFace Standard (Normalize and reshape)
        input_img = aligned_face.astype(np.float32)
        input_img = (input_img - 127.5) / 127.5
        input_img = input_img.transpose(2, 0, 1) # HWC to CHW
        input_tensor = np.expand_dims(input_img, axis=0) # Add batch dimension (1, 3, 112, 112)
        
        # Inference
        rec_output = self.rec_session.run(None, {self.rec_input_name: input_tensor})
        embedding = rec_output[0][0]
        
        # Normalization (Essential for Cosine Similarity)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding