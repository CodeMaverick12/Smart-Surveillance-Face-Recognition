
import cv2
import os
import logging
from src.detection.face_detector import FaceDetector
from src.utilis.video_utils import open_video_source, extract_frames, display_video, capture_single_frame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    logging.info("🚀 Starting Smart Surveillance System...")

    # Initialize FaceDetector
    # You can choose 'haar', 'ssd', or 'mtcnn'
    face_detector = FaceDetector(model_name="mtcnn", detector_backend="mtcnn")
    logging.info(f"✅ FaceDetector initialized with model: {face_detector.model_name}")

    # --- Option 1: Live Webcam Feed with Face Detection ---
    # Uncomment to run live detection
    print("\n--- Live Webcam Face Detection (Press 'q' to quit) ---")
    cap = open_video_source(source=0) # Use 0 for webcam
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("⚠️ Failed to grab frame from video source. Exiting live feed.")
            break

        detected_faces = face_detector.detect_faces(frame)
        if isinstance(detected_faces, list):
            for face_data in detected_faces:
                if face_detector.model_name == "mtcnn":
                    # For MTCNN via DeepFace, face_data is a dict with 'box' key
                    x, y, w, h = face_data['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    # For Haar and SSD, face_data is a tuple (x, y, w, h)
                    x, y, w, h = face_data
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.imshow("Live Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("🛑 'q' pressed. Exiting live face detection.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Live webcam face detection ended.")

    # --- Option 2: Process a Video File (Example) ---
    # video_file_path = "path/to/your/video.mp4" # Replace with your video file
    # if os.path.exists(video_file_path):
    #     print("\n--- Processing Video File with Face Detection ---")
    #     cap_file = open_video_source(source=video_file_path)
    #     if cap_file:
    #         frame_idx = 0
    #         while True:
    #             ret, frame = cap_file.read()
    #             if not ret:
    #                 logging.info("⚠️ End of video file or stream not available.")
    #                 break
    #
    #             if frame_idx % 30 == 0: # Process every 30th frame for efficiency
    #                 detected_faces_file = face_detector.detect_faces(frame)
    #                 for (x, y, w, h) in detected_faces_file:
    #                     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #                 logging.info(f"Processed frame {frame_idx}: {len(detected_faces_file)} faces detected.")
    #
    #                 cv2.imshow("Video File Face Detection", frame)
    #                 if cv2.waitKey(1) & 0xFF == ord('q'):
    #                     logging.info("🛑 'q' pressed. Exiting video file processing.")
    #                     break
    #             frame_idx += 1
    #
    #         cap_file.release()
    #         cv2.destroyAllWindows()
    #         logging.info("Video file processing ended.")
    # else:
    #     logging.warning(f"Video file not found: {video_file_path}. Skipping video file processing.")

    logging.info("✅ Smart Surveillance System finished.")

if __name__ == "__main__":
    main()
