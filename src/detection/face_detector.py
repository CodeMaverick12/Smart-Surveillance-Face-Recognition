import cv2
import numpy as np
import os
from deepface import DeepFace

class FaceDetector:
    """
    FaceDetector class handles detection and embedding extraction of faces
    from a given video frame using pre-trained models (HaarCascade, SSD, or MTCNN).
    """

    def __init__(self, model_name="mtcnn", detector_backend="opencv"):
        """
        Initialize the FaceDetector with chosen backend.
        Args:
            model_name (str): Face detection model ('mtcnn', 'ssd', 'haar').
            detector_backend (str): Backend used by DeepFace for embedding extraction.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.face_detector = self._load_detector()

    def _load_detector(self):
        """Load the face detection model based on chosen type."""
        if self.model_name.lower() == "haar":
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif self.model_name.lower() == "ssd":
            # Load pre-trained SSD model
            prototxt = "models/deploy.prototxt"
            weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
            if not os.path.exists(prototxt) or not os.path.exists(weights):
                raise FileNotFoundError("SSD model files missing in /models directory.")
            return cv2.dnn.readNetFromCaffe(prototxt, weights)
        elif self.model_name.lower() == "mtcnn":
            # MTCNN is internally supported by DeepFace
            return None
        else:
            raise ValueError("Unsupported detector model. Use 'haar', 'ssd', or 'mtcnn'.")

    def detect_faces(self, frame):
        """
        Detect faces in a given frame and return bounding boxes.
        Returns:
            List of (x, y, w, h) tuples
        """
        faces = []

        if self.model_name == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_detector.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in detected]

        elif self.model_name == "ssd":
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        elif self.model_name == "mtcnn":
            # DeepFace will internally handle MTCNN detection
            try:
                faces = DeepFace.extract_faces(img_path=frame, detector_backend='mtcnn', enforce_detection=False)
            except Exception as e:
                print(f"[Error] MTCNN detection failed: {e}")
                faces = []

        return faces

    def get_face_embeddings(self, face_image):
        """
        Extracts the facial embedding vector using DeepFace.
        Args:
            face_image: cropped face image (numpy array)
        Returns:
            Embedding vector (numpy array)
        """
        try:
            embedding = DeepFace.represent(face_image, model_name="Facenet", detector_backend=self.detector_backend, enforce_detection=False)
            return np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"[Error] Embedding extraction failed: {e}")
            return None

    def extract_faces_from_frame(self, frame):
        """
        Detect and crop faces from a frame.
        Returns:
            List of cropped face images.
        """
        boxes = self.detect_faces(frame)
        face_images = []

        for (x, y, w, h) in boxes:
            face = frame[y:y+h, x:x+w]
            face_images.append(face)

        return face_images
