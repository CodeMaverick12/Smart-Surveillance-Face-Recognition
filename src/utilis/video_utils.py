import sys
import os
import cv2
import numpy as np
import json

# --- GUARANTEED PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)

# ✅ Now import SORT after adding src to path
from utilis.sort import Sort
from detection.face_detector import FaceDetector

# Global Constants
EMBEDDING_DIM = 512
RECOGNITION_THRESHOLD = 0.85  # Cosine distance threshold (0.6-1.0 range)
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 1000


# --- Identity Database Class (Simple Cosine Similarity) ---
class IdentityDatabase:
    def __init__(self, dim=EMBEDDING_DIM, save_dir='data/identities'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.embeddings_path = os.path.join(self.save_dir, 'embeddings.npy')
        self.metadata_path = os.path.join(self.save_dir, 'identity_metadata.json')
        
        self.embeddings = []  # List of stored embeddings
        self.embedding_to_person = []  # Maps each embedding to person_id
        self.next_id = 0
        self.id_map = {}
        
        self.load_database()

    def save_database(self):
        """Save embeddings and metadata to disk"""
        try:
            if len(self.embeddings) > 0:
                np.save(self.embeddings_path, np.array(self.embeddings))
            
            metadata = {
                'next_id': self.next_id,
                'id_map': self.id_map,
                'embedding_to_person': self.embedding_to_person
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Saved {len(self.embeddings)} embeddings for {self.next_id} people")
            
        except Exception as e:
            print(f"❌ Error saving database: {e}")

    def load_database(self):
        """Load existing embeddings and metadata from disk"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.next_id = metadata['next_id']
                self.id_map = {int(k): v for k, v in metadata['id_map'].items()}
                self.embedding_to_person = metadata.get('embedding_to_person', [])
                
                if os.path.exists(self.embeddings_path):
                    self.embeddings = np.load(self.embeddings_path).tolist()
                    print(f"✅ Loaded {len(self.embeddings)} embeddings for {self.next_id} people")
                else:
                    print("📝 No embeddings found. Starting fresh.")
            else:
                print("📝 Starting new identity database.")
                
        except Exception as e:
            print(f"❌ Error loading database: {e}")

    def search_identity(self, embedding):
        """Search for matching identity using cosine similarity"""
        if len(self.embeddings) == 0:
            return None
        
        # Calculate cosine similarity with all stored embeddings
        embedding_norm = embedding / np.linalg.norm(embedding)
        stored_embeddings = np.array(self.embeddings)
        
        # Normalize stored embeddings
        norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
        stored_embeddings_norm = stored_embeddings / norms
        
        # Compute similarities
        similarities = np.dot(stored_embeddings_norm, embedding_norm)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Convert similarity to distance
        distance = 2 * (1 - best_similarity)
        
        if distance < RECOGNITION_THRESHOLD:
            person_idx = self.embedding_to_person[best_idx]
            matched_id = self.id_map[person_idx]['person_id']
            print(f"✅ Recognized: {matched_id} (distance: {distance:.3f})")
            return matched_id
        
        return None

    def add_embedding_to_person(self, person_idx, embedding):
        """Add additional embedding for existing person"""
        self.embeddings.append(embedding.tolist())
        self.embedding_to_person.append(person_idx)
        self.save_database()

    def register_new_identity(self, embedding, age, gender):
        """Register a new identity"""
        person_idx = self.next_id
        
        # Add first embedding for this person
        self.embeddings.append(embedding.tolist())
        self.embedding_to_person.append(person_idx)
        
        person_id = f"Person_{self.next_id:04d}"
        self.id_map[self.next_id] = {
            'person_id': person_id, 
            'age': age, 
            'gender': gender,
            'embedding_count': 1
        }
        self.next_id += 1
        
        self.save_database()
        return person_id

    def update_person_embeddings(self, person_id, embedding, max_embeddings=5):
        """Add new embedding to existing person (up to max limit)"""
        # Find person index
        person_idx = None
        for idx, data in self.id_map.items():
            if data['person_id'] == person_id:
                person_idx = idx
                break
        
        if person_idx is None:
            return
        
        # Count current embeddings for this person
        current_count = self.id_map[person_idx].get('embedding_count', 1)
        
        # Add new embedding if under limit
        if current_count < max_embeddings:
            self.embeddings.append(embedding.tolist())
            self.embedding_to_person.append(person_idx)
            self.id_map[person_idx]['embedding_count'] = current_count + 1
            print(f"📸 Added embedding #{current_count + 1} for {person_id}")
            self.save_database()


# --- Helper: IOU Calculation ---
def calculate_iou(boxA, boxB):
    x1, y1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    x2, y2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# --- Main Video Processing Function ---
# --- Main Video Processing Function with ID Smoothing ---
def process_video_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file {video_path}")
        return

    processor = FaceDetector()
    id_db = IdentityDatabase()
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    track_to_person = {}
    frame_count = {}
    
    # ✅ NEW: Track ID confidence and history
    track_id_history = {}  # track_id -> list of recent person_ids
    track_id_confidence = {}  # track_id -> confidence score
    HISTORY_SIZE = 10  # Number of frames to consider for smoothing
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to change ID

    print("🎥 Processing video feed... Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces_data = processor.get_face_embeddings(frame)
            detections = []

            for face in faces_data:
                (x1, y1, x2, y2) = face['bbox']
                detections.append([x1, y1, x2, y2, 1.0])

            detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
            tracks = tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_bbox = (int(x1), int(y1), int(x2), int(y2))
                track_id = int(track_id)

                best_match_idx = None
                best_iou = 0.3

                for idx, face_data in enumerate(faces_data):
                    iou = calculate_iou(track_bbox, face_data['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx

                if best_match_idx is not None:
                    face_data = faces_data[best_match_idx]
                    embedding = face_data['embedding']

                    # Search for identity
                    detected_person_id = id_db.search_identity(embedding)
                    
                    # ✅ NEW: Initialize history for new tracks
                    if track_id not in track_id_history:
                        track_id_history[track_id] = []
                        track_id_confidence[track_id] = 0.0
                    
                    # ✅ NEW: Handle ID assignment with confidence
                    if detected_person_id is not None:
                        # Found a match - add to history
                        track_id_history[track_id].append(detected_person_id)
                        
                        # Keep only recent history
                        if len(track_id_history[track_id]) > HISTORY_SIZE:
                            track_id_history[track_id].pop(0)
                        
                        # Calculate confidence (most common ID in history)
                        from collections import Counter
                        id_counts = Counter(track_id_history[track_id])
                        most_common_id, count = id_counts.most_common(1)[0]
                        confidence = count / len(track_id_history[track_id])
                        
                        # ✅ NEW: Only update ID if confidence is high enough
                        if track_id in track_to_person:
                            current_id = track_to_person[track_id]
                            # If new ID has higher confidence, switch
                            if most_common_id != current_id and confidence >= CONFIDENCE_THRESHOLD:
                                track_to_person[track_id] = most_common_id
                                track_id_confidence[track_id] = confidence
                                print(f"🔄 Updated ID for track {track_id}: {current_id} -> {most_common_id} (confidence: {confidence:.2f})")
                            else:
                                # Keep current ID
                                person_id = current_id
                        else:
                            # New track - assign ID
                            track_to_person[track_id] = most_common_id
                            track_id_confidence[track_id] = confidence
                        
                        person_id = track_to_person[track_id]
                        
                        # Update embeddings periodically
                        if person_id not in frame_count:
                            frame_count[person_id] = 0
                        frame_count[person_id] += 1
                        
                        if frame_count[person_id] % 30 == 0:
                            id_db.update_person_embeddings(person_id, embedding)
                        
                    elif track_id in track_to_person:
                        # ✅ No match found, but track exists - keep using existing ID
                        person_id = track_to_person[track_id]
                        # Add current ID to history to maintain continuity
                        track_id_history[track_id].append(person_id)
                        if len(track_id_history[track_id]) > HISTORY_SIZE:
                            track_id_history[track_id].pop(0)
                        
                    else:
                        # ✅ New person - register
                        person_id = id_db.register_new_identity(
                            embedding,
                            face_data['age'],
                            face_data['gender']
                        )
                        track_to_person[track_id] = person_id
                        track_id_history[track_id] = [person_id]
                        track_id_confidence[track_id] = 1.0
                        frame_count[person_id] = 1
                        print(f"🆕 Registered new identity: {person_id}")

                    # Draw results
                    label = f"{person_id}"
                    
                    # ✅ Optional: Show confidence
                    if track_id in track_id_confidence:
                        conf = track_id_confidence[track_id]
                        label = f"{person_id} ({conf:.0%})"
                    
                    cv2.rectangle(frame, (track_bbox[0], track_bbox[1]),
                                 (track_bbox[2], track_bbox[3]), (0, 255, 0), 2)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    
                    cv2.rectangle(frame, (track_bbox[0], track_bbox[1] - text_h - 10),
                                 (track_bbox[0] + text_w, track_bbox[1]), (0, 0, 0), -1)
                    cv2.putText(frame, label, (track_bbox[0], track_bbox[1] - 7),
                               font, font_scale, (255, 255, 255), font_thickness)

            # Clean up old tracks
            active_track_ids = set([int(t[4]) for t in tracks])
            track_to_person = {k: v for k, v in track_to_person.items() if k in active_track_ids}
            track_id_history = {k: v for k, v in track_id_history.items() if k in active_track_ids}
            track_id_confidence = {k: v for k, v in track_id_confidence.items() if k in active_track_ids}

            # Display
            (h, w) = frame.shape[:2]
            scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
            display_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            cv2.imshow('Smart Surveillance (Stable ID)', display_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    finally:
        print("\n💾 Saving database before exit...")
        id_db.save_database()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Database saved successfully!")
# --- Main Entry ---
if __name__ == '__main__':
    user_input = input("Enter video file path (or type 'live' for webcam): ").strip()

    if user_input.lower() == 'live':
        INPUT_SOURCE = 0
    elif os.path.exists(user_input):
        INPUT_SOURCE = user_input
    else:
        INPUT_SOURCE = 'output_tracked_video (1).mp4'
        if not os.path.exists(INPUT_SOURCE):
            print("❌ No valid source found. Exiting.")
            sys.exit(1)

    process_video_feed(INPUT_SOURCE)