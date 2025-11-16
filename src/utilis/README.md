# Video Utilities for Smart Surveillance System

This module provides functionality for **video processing, face recognition, and identity tracking**. It integrates face embeddings from `FaceDetector` and uses the SORT tracker to maintain **stable IDs across frames**.

## Features

- Processes video feeds (live webcam or video files) for face recognition.
- Integrates with the `IdentityDatabase` for registering and recognizing identities.
- Implements **ID smoothing** for stable identity tracking over multiple frames.
- Uses **SORT algorithm** (`sort.py`) for robust tracking.
- Displays bounding boxes and identity labels on video frames.
- Optional: Updates embeddings periodically to improve recognition.

## Key Classes and Functions

### `IdentityDatabase`
- Stores embeddings and metadata of known identities.
- Functions:
  - `search_identity(embedding)` → Returns recognized person ID.
  - `register_new_identity(embedding, age, gender)` → Adds new person.
  - `update_person_embeddings(person_id, embedding)` → Adds more embeddings for existing person.

### `process_video_feed(video_path)`
- Main video processing function.
- Steps:
  1. Read video frames.
  2. Detect faces and extract embeddings using `FaceDetector`.
  3. Track faces using SORT.
  4. Match embeddings with `IdentityDatabase`.
  5. Draw bounding boxes and labels on frames.
  6. Display video and save database on exit.

### `calculate_iou(boxA, boxB)`
- Computes Intersection-over-Union (IoU) between two bounding boxes.
- Used internally by SORT for matching detections to trackers.

## Usage Example

```python
from video_utils import process_video_feed

# Process a video file
process_video_feed("example_video.mp4")

# Use live webcam feed
process_video_feed(0)
