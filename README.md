
# Smart Surveillance Face Recognition & Re-Identification

This project aims to develop a smart surveillance system capable of real-time face recognition and re-identification for the Pakistani community. It leverages computer vision techniques to detect faces, extract unique facial embeddings, and match them against a database.

## Project Structure

```
.Smart-Surveillance-Face-Recognition-Re-Identification-for-Pakistani-Community/
├── data/
│   ├── faces/        # Stores cropped and aligned faces
│   └── frames/       # Stores extracted video frames
├── src/
│   ├── detection/
│   │   ├── face_alignment.py
│   │   ├── face_detector.py
│   │   └── README.md
│   └── utilis/
│       ├── video_utils.py
│       └── README.md
|       └── sort.py
├── models/           # Directory to store pre-trained models (e.g., SSD, MTCNN weights)
├── main.py           # Main script to run the surveillance system
└── README.md         # Project-level documentation
```

## Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

Ensure you have Python 3.x installed. You will also need `pip` for package installation.

### 1. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 2. Install Dependencies

Install the required Python packages. You will need `opencv-python`, `numpy`, and `deepface`. It's recommended to create a `requirements.txt` file.

First, let's create a `requirements.txt` file.

```bash
# Create requirements.txt
pip freeze > requirements.txt
```

Then, install them:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the main dependencies:

```bash
pip install opencv-python numpy deepface
```

### 3. Download Models (for SSD Face Detector)

If you plan to use the SSD face detection model, you need to download its pre-trained files and place them in a `models/` directory at the project root.

-   `deploy.prototxt`: [https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/bvlc_googlenet.prototxt](https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/bvlc_googlenet.prototxt)
-   `res10_300x300_ssd_iter_140000.caffemodel`: https://huggingface.co/spaces/liangtian/birthdayCrown/blob/3db8f1c391e44bd9075b1c2854634f76c2ff46d0/res10_300x300_ssd_iter_140000.caffemodel
-   `arcface_r50.onnx`:
https://huggingface.co/facefusion/models-3.0.0/blob/main/arcface_w600k_r50.onnx

Create a `models` directory:

```bash
mkdir models
```

Then place the downloaded `.prototxt` and `.caffemodel` ,`.onnx`files into this `models` directory.

For `mtcnn` or `haar` detection, DeepFace handles model downloads automatically or OpenCV provides Haar cascades, so no manual download is typically needed for those.

### 4. Run the Main Application

To run the live webcam face detection:

```bash
python main.py
```

Press 'q' to quit the live video feed.

## `main.py` Overview

-   Initializes the `FaceDetector` (defaulting to `mtcnn`).
-   Opens the default webcam (source=0).
-   Continuously reads frames, detects faces using the chosen model, and draws bounding boxes around them.
-   Displays the live feed with detected faces.

## Module Documentation

Detailed documentation for individual modules can be found in their respective `README.md` files:

-   [`src/utilis/README.md`](src/utilis/README.md): Documentation for video utility functions.
-   [`src/detection/README.md`](src/detection/README.md): Documentation for the face detection class.


