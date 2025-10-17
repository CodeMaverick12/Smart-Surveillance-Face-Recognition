"""
video_utils.py
---------------
Handles video feed input, frame extraction, and saving frames for face detection.
Supports both live camera streams and prerecorded videos.
"""

import cv2
import os
import time
import logging


# ---------------------- LOGGER CONFIG ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------------- VIDEO UTILS ----------------------
def open_video_source(source=0):
    """
    Opens a video source (default: webcam index 0).
    You can pass a video file path or RTSP/HTTP stream URL as well.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"❌ Unable to open video source: {source}")
        raise ValueError(f"Could not open video source {source}")
    logging.info(f"✅ Video source opened: {source}")
    return cap


def extract_frames(video_source=0, output_dir="../../data/frames", skip_frames=5, limit=None):
    """
    Extracts frames from a video feed or file.
    Args:
        video_source: int | str -> camera index or video file path.
        output_dir: directory to save extracted frames.
        skip_frames: number of frames to skip between extractions.
        limit: optional max number of frames to extract (for testing).
    Returns:
        list of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = open_video_source(video_source)

    saved_frames = []
    frame_count = 0
    saved_count = 0

    logging.info("🎥 Starting frame extraction...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("⚠️ End of video or stream not available.")
            break

        if frame_count % skip_frames == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_frames.append(filename)
            saved_count += 1

            logging.info(f"📸 Saved frame {saved_count} → {filename}")

            if limit and saved_count >= limit:
                logging.info(f"🛑 Frame limit reached: {limit}")
                break

        frame_count += 1

    cap.release()
    logging.info(f"✅ Frame extraction complete. Total saved: {saved_count}")
    return saved_frames


def display_video(video_source=0, window_name="Video Feed"):
    """
    Displays live video feed for testing.
    Press 'q' to quit.
    """
    cap = open_video_source(video_source)
    logging.info("🔴 Press 'q' to exit live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("🟢 Video display ended.")


def capture_single_frame(video_source=0, output_path="../../data/frames/snapshot.jpg"):
    """
    Captures a single frame from video or webcam and saves it.
    Useful for testing the detection pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = open_video_source(video_source)

    ret, frame = cap.read()
    if not ret:
        logging.error("❌ Unable to capture frame.")
        return None

    cv2.imwrite(output_path, frame)
    logging.info(f"📷 Snapshot saved to {output_path}")

    cap.release()
    return output_path


# ---------------------- TESTING (if run directly) ----------------------
if __name__ == "__main__":
    # Test the display_video function with webcam (source=0)
    # video_path = './ReliefHub - Google Chrome 2025-09-29 23-54-16.mp4'
    # frames = extract_frames(video_source=0, skip_frames=10, limit=5)
    # print(f"Extracted {len(frames)} frames.")

    print("\n--- Testing display_video with webcam ---")
    display_video(video_source=0, window_name="Webcam Feed")

    print("\n--- Testing capture_single_frame with webcam ---")
    snapshot_path = capture_single_frame(video_source=0)
    if snapshot_path:
        print(f"Captured snapshot to {snapshot_path}")

    print("\n--- Testing extract_frames with webcam (5 frames, skip 10) ---")
    extracted_frames = extract_frames(video_source=0, output_dir="../../data/frames", skip_frames=10, limit=5)
    print(f"Extracted {len(extracted_frames)} frames: {extracted_frames}")

    print("\n--- All video_utils tests completed ---")
