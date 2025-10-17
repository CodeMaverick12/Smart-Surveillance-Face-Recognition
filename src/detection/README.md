
# `face_detector.py` Documentation

## Purpose

The `face_detector.py` file defines the `FaceDetector` class, which is a core component for face detection and embedding extraction within the Smart Surveillance system. It provides functionalities to identify faces in video frames using various algorithms (Haar Cascade, SSD, MTCNN) and to generate numerical representations (embeddings) of these faces for subsequent recognition and re-identification tasks.

## Class: `FaceDetector`

### Description

The `FaceDetector` class encapsulates the logic for detecting faces and extracting their facial embeddings from video frames. It offers flexibility by supporting multiple underlying face detection models.

### `__init__(self, model_name="mtcnn", detector_backend="opencv")`

-   **Description:** Initializes the `FaceDetector` instance, setting up the desired face detection model and the backend for DeepFace operations.
-   **Arguments:**
    -   `model_name` (str, optional): Specifies the face detection model to be used. Accepted values are `'haar'`, `'ssd'`, or `'mtcnn'` (default is `'mtcnn'`)
    -   `detector_backend` (str, optional): Defines the backend that DeepFace will use for extracting facial embeddings. Common choices include `'opencv'`, `'ssd'`, `'mtcnn'`, `'retinaface'`, `'mediapipe'`, `'yolov8'` (default is `'opencv'`).
-   **Flow:**
    1.  Stores the `model_name` and `detector_backend` as instance attributes.
    2.  Calls the private method `_load_detector()` to load and initialize the specified face detection model, which varies based on the `model_name`.

### `_load_detector(self)` (Private Method)

-   **Description:** This internal method is responsible for loading the appropriate face detection model based on the `model_name` provided during the class initialization.
-   **Returns:**
    -   `cv2.CascadeClassifier` object if `model_name` is `'haar'`.
    -   `cv2.dnn.Net` object if `model_name` is `'ssd'`.
    -   `None` if `model_name` is `'mtcnn'` (as MTCNN is handled internally by DeepFace).
-   **Raises:**
    -   `FileNotFoundError`: If required SSD model files (`.prototxt`, `.caffemodel`) are missing.
    -   `ValueError`: If an unsupported `model_name` is provided.
-   **Flow:**
    1.  Checks `self.model_name`:
        -   If `'haar'`, it loads the `haarcascade_frontalface_default.xml` classifier.
        -   If `'ssd'`, it loads the pre-trained Caffe-based SSD model from specified `prototxt` and `weights` files. It verifies their existence.
        -   If `'mtcnn'`, it returns `None` because the DeepFace library handles MTCNN model loading and inference automatically.
        -   For any other `model_name`, it raises a `ValueError`.

### `detect_faces(self, frame)`

-   **Description:** Detects faces within a given input video frame and returns a list of bounding boxes corresponding to the detected faces.
-   **Arguments:**
    -   `frame` (numpy.ndarray): The input image frame (in BGR format).
-   **Returns:** `list` of `tuple`, where each tuple `(x, y, w, h)` represents the top-left corner coordinates (`x`, `y`) and the width (`w`) and height (`h`) of a detected face's bounding box.
-   **Flow:**
    1.  Initializes an empty list `faces`.
    2.  Based on `self.model_name`:
        -   **Haar Cascade:** Converts the frame to grayscale and applies `self.face_detector.detectMultiScale()` to find faces.
        -   **SSD:** Resizes the frame to 300x300, creates a blob, sets it as input to the loaded DNN model, performs a forward pass, and then filters detections based on a confidence threshold (0.5). It scales the bounding box coordinates back to the original frame size.
        -   **MTCNN:** Utilizes `DeepFace.extract_faces()` with `detector_backend='mtcnn'` to detect faces. Includes a `try-except` block to catch and print errors during MTCNN detection.
    3.  Returns the list of detected `faces`.

### `get_face_embeddings(self, face_image)`

-   **Description:** Extracts a 512-dimensional facial embedding vector from a cropped face image using the `Facenet` model via DeepFace. These embeddings are crucial for comparing faces and performing re-identification.
-   **Arguments:**
    -   `face_image` (numpy.ndarray): A NumPy array representing a cropped image of a single face.
-   **Returns:** `numpy.ndarray` (a 512-element vector) representing the facial embedding, or `None` if the embedding extraction fails.
-   **Flow:**
    1.  Calls `DeepFace.represent()` with the `face_image`, specifying `model_name="Facenet"` and the `detector_backend` set during initialization.
    2.  Extracts the embedding vector from the DeepFace result.
    3.  Includes a `try-except` block to handle and print any errors that occur during embedding extraction.

### `extract_faces_from_frame(self, frame)`

-   **Description:** This method combines face detection and cropping. It first detects all faces in an input frame and then returns a list of individual cropped face images.
-   **Arguments:**
    -   `frame` (numpy.ndarray): The input image frame.
-   **Returns:** `list` of `numpy.ndarray`, where each array is a cropped image of a detected face.
-   **Flow:**
    1.  Calls `self.detect_faces(frame)` to obtain the bounding boxes of all detected faces.
    2.  Iterates through each bounding box.
    3.  For each bounding box, it crops the corresponding face region from the original `frame`.
    4.  Appends the cropped face image to a list.
    5.  Returns the list of cropped `face_images`.

## Overall Flow

The `face_detector.py` file provides a robust and flexible solution for integrating face detection and embedding extraction into the surveillance system. The `FaceDetector` class acts as a central interface, allowing users to select different detection models based on their needs (e.g., speed vs. accuracy). The workflow typically involves:

1.  **Initialization:** Creating a `FaceDetector` instance, specifying the desired `model_name` and `detector_backend`.
2.  **Face Detection:** Calling `detect_faces()` on a video frame to get bounding box coordinates.
3.  **Face Cropping:** Using `extract_faces_from_frame()` to obtain cropped images of detected faces.
4.  **Embedding Extraction:** Applying `get_face_embeddings()` on cropped face images to generate their unique facial representations. These embeddings are then used for comparison and re-identification.

This modular design allows for easy swapping of detection and embedding models and streamlined integration into higher-level facial recognition pipelines.
