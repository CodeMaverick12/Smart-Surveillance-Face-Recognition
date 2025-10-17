
# `video_utils.py` Documentation

## Purpose

The `video_utils.py` file is designed to handle various video-related operations within the Smart Surveillance system. Its primary functions include managing video feed input, extracting frames from video sources, and saving these frames for further processing, such as face detection. The utility supports both live camera streams and pre-recorded video files.

## Functions

### `open_video_source(source=0)`

-   **Description:** Opens a video source, which can be a webcam (by index) or a video file/stream URL.
-   **Arguments:**
    -   `source` (int | str, optional): The camera index (default is 0) or the path/URL to a video file/stream.
-   **Returns:** `cv2.VideoCapture` object. This object is used by other functions to read video frames.
-   **Flow:**
    1.  Initializes `cv2.VideoCapture` with the provided `source`.
    2.  Checks if the video source was successfully opened. If not, it logs an error and raises a `ValueError`.
    3.  Logs a success message and returns the `cv2.VideoCapture` object.

### `extract_frames(video_source=0, output_dir="../../data/frames", skip_frames=5, limit=None)`

-   **Description:** Extracts frames from a specified video source and saves them to a designated output directory. This function can be configured to skip frames and limit the total number of frames extracted.
-   **Arguments:**
    -   `video_source` (int | str, optional): The camera index or video file path (default is 0).
    -   `output_dir` (str, optional): The directory where the extracted frames will be saved (default is `../../data/frames`).
    -   `skip_frames` (int, optional): The number of frames to skip between each extraction. A value of 1 means every frame is extracted, 5 means every 5th frame, etc. (default is 5).
    -   `limit` (int | None, optional): An optional maximum number of frames to extract. Useful for testing or limiting resource usage (default is `None`, meaning no limit).
-   **Returns:** `list` of strings, where each string is the file path of a saved frame.
-   **Flow:**
    1.  Ensures the `output_dir` exists, creating it if necessary.
    2.  Opens the `video_source` using `open_video_source()`.
    3.  Enters a loop to continuously read frames from the video source.
    4.  If a frame is successfully read and the `frame_count` meets the `skip_frames` interval, the frame is saved as a JPG image in `output_dir`.
    5.  The process stops if the end of the video stream is reached or the `limit` for saved frames is met.
    6.  Releases the `cv2.VideoCapture` object and logs the completion of frame extraction.

### `display_video(video_source=0, window_name="Video Feed")`

-   **Description:** Displays a live video feed in a pop-up window, primarily used for testing and real-time monitoring. Users can press 'q' to quit the display.
-   **Arguments:**
    -   `video_source` (int | str, optional): The camera index or video file path (default is 0).
    -   `window_name` (str, optional): The title of the display window (default is "Video Feed").
-   **Returns:** `None`.
-   **Flow:**
    1.  Opens the `video_source` using `open_video_source()`.
    2.  Enters a loop to read and display frames using `cv2.imshow()`.
    3.  The loop continues until the 'q' key is pressed or the video stream ends.
    4.  Releases the `cv2.VideoCapture` object and destroys all OpenCV windows.

### `capture_single_frame(video_source=0, output_path="../../data/frames/snapshot.jpg")`

-   **Description:** Captures a single frame from the specified video source or webcam and saves it to a designated file path. This is useful for capturing quick snapshots for testing the detection pipeline.
-   **Arguments:**
    -   `video_source` (int | str, optional): The camera index or video file path (default is 0).
    -   `output_path` (str, optional): The full path, including filename, where the captured snapshot will be saved (default is `../../data/frames/snapshot.jpg`).
-   **Returns:** `str` (the path to the saved frame) if successful, otherwise `None`.
-   **Flow:**
    1.  Ensures the directory for `output_path` exists.
    2.  Opens the `video_source` using `open_video_source()`.
    3.  Attempts to read a single frame. If successful, the frame is saved to `output_path`.
    4.  Logs a success or error message based on the capture outcome.
    5.  Releases the `cv2.VideoCapture` object.

## Overall Flow

The `video_utils.py` file acts as a foundational utility for video handling. It provides modular functions that can be independently called or integrated into a larger system. The `open_video_source` function serves as a core component, ensuring proper video stream initialization for all other functions. The primary operations revolve around capturing, extracting, and displaying video frames, making it suitable for pre-processing video data for tasks like face recognition and re-identification. The `if __name__ == "__main__":` block demonstrates a basic usage example by extracting a limited number of frames from the default webcam.
