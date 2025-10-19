# src/utilis/video_utils.py - FINAL WORKING CODE FOR DAY 2

import sys
import os
import cv2
import faiss
import numpy as np

# --- GUARANTEED PATH FIX ---
# Add the 'src' directory (the parent of both 'detection' and 'utilis') to the Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the 'src' directory
src_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.insert(0, src_dir)

from detection.face_detector import FaceProcessor 

# Global Constants
EMBEDDING_DIM = 512
# UPDATED: Increased threshold for better recognition robustness (less flickering IDs)
RECOGNITION_THRESHOLD = 3.0
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 1000 # Global display width for resizing the output window

# --- SCALABILITY: Faiss IndexIVFPQ Implementation ---
# src/utilis/video_utils.py - CORRECTED IdentityDatabase
# src/utilis/video_utils.py - IdentityDatabase.__init__ FIX

class IdentityDatabase:
    def __init__(self, dim=EMBEDDING_DIM):
        quantizer = faiss.IndexFlatL2(dim)
        
        nlist = 40    
        m = 8         
        nbits = 8     
        
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        
        # --- CRITICAL FIX: ADD THESE ATTRIBUTES ---
        self.next_id = 0
        self.id_map = {} 
        self.trained = False  # <-- THIS WAS MISSING OR MISPLACED
        # ------------------------------------------

        self.train_data = [] # Temporary list to collect vectors for training

    def search_identity(self, embedding):
        if not self.trained:
            # If not trained, fall back to simple brute-force check on collected data (temporary fix)
            if not self.train_data:
                return None
            
            # For simplicity during training phase, we skip search and force registration of new data
            # to collect the 100 vectors needed for training.
            return None 
            
        # The index is trained, proceed with scalable search
        self.index.nprobe = 10
        D, I = self.index.search(embedding.reshape(1, -1), 1)
        
        distance = D[0][0]
        faiss_id = I[0][0]

        if faiss_id != -1 and distance < RECOGNITION_THRESHOLD:
            # Faiss IDs correspond to our self.next_id counter
            return self.id_map[faiss_id]['person_id']
        else:
            return None
    def register_new_identity(self, embedding, age, gender):
        # Training/Adding Logic
        
        if not self.trained:
            # 1. Collect data for training
            self.train_data.append(embedding)
            
            # FIX 2: INCREASE TRAINING SIZE for better clustering
            if len(self.train_data) >= 300: # <-- CHANGED from 100 to 300
                print("--- TRAINING FAISS INDEX (300 vectors) --- This may take a moment...")
                train_matrix = np.vstack(self.train_data).astype('float32')
                self.index.train(train_matrix)
                self.trained = True
                print("Faiss Index is now TRAINED and SCALABLE (IndexIVFPQ)!")
                
                # 2. Add the initial collected vectors to the trained index
                ids = np.arange(0, len(self.train_data)).astype(np.int64)
                self.index.add_with_ids(train_matrix, ids)
                print(f"Added {len(self.train_data)} initial vectors to the index.")
                self.train_data = [] # Clear training data to save memory
        
        # ... rest of the registration logic ...
        # (This section remains the same, adding the vector if trained)
        if self.trained:
             self.index.add_with_ids(embedding.reshape(1, -1).astype('float32'), np.array([self.next_id], dtype=np.int64))

        person_id = f"Person_{self.next_id:04d}"
        self.id_map[self.next_id] = {'person_id': person_id, 'age': age, 'gender': gender}
        self.next_id += 1
        return person_id
def process_video_feed(video_path):
    """Main processing loop for the video feed."""
    
    # --- CRITICAL FIXES FOR NAME ERRORS ---
    global last_tracked_faces # Ensure we can modify the tracker list
    
    # 1. Initialize Video Capture (cap)
    cap = cv2.VideoCapture(video_path) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 2. Initialize Core Modules (processor and id_db)
    processor = FaceProcessor()
    id_db = IdentityDatabase()
    
    # Clear the global tracker before starting the video processing loop
    last_tracked_faces = [] 
    
    # --- END OF CRITICAL FIXES ---
    
    # The rest of the while loop starts here:
    while cap.isOpened():
    
        ret, frame = cap.read()
        if not ret: break

        faces_data = processor.get_face_embeddings(frame)
        current_tracked_faces = [] # Tracks faces in the CURRENT frame
        
        for data in faces_data:
            bbox = data['bbox']
            embedding = data['embedding']
            
            person_id = None
            best_iou = 0

            # 1. CHECK HISTORY (IoU Tracking): Avoids Faiss search if face hasn't moved much
            for (last_bbox, last_id) in last_tracked_faces:
                iou = calculate_iou(bbox, last_bbox)
                if iou > best_iou and iou > MAX_IOU_DISTANCE:
                    person_id = last_id # Re-use the previous ID
                    best_iou = iou
            
            if person_id is None:
                # 2. NEW/UNTRACKED FACE: Perform Faiss DB search
                person_id = id_db.search_identity(embedding)

                if person_id is None:
                    # 3. REGISTER NEW ID (ONLY if no match in history AND no match in Faiss)
                    person_id = id_db.register_new_identity(embedding, data['age'], data['gender'])
            # src/utilis/video_utils.py - Inside the main processing loop

            # ... (Logic that determines person_id is above this line) ...

            # 4. Store for the next frame's tracking check
            # This line correctly uses the determined person_id
            current_tracked_faces.append((bbox, person_id)) 
            
            # 5. Draw Results (Using the local variable 'person_id')
            (x1, y1, x2, y2) = bbox
            
            # --- CRITICAL FIX: DO NOT ACCESS data['person_id'] ---
            # Just use the local variable 'person_id' defined earlier in the loop
            label = f"{person_id}" # <-- THIS USES THE CORRECT LOCAL VARIABLE
            
            # --- Draw the solid background and text for clear ID visibility ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # Calculate the size of the text bounding box
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw solid BLACK background rectangle 
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1) 
            
            # Draw the bright GREEN Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), font_thickness)
            
            # Draw the WHITE Text Label
            cv2.putText(frame, label, (x1, y1 - 7), 
                        font, font_scale, (255, 255, 255), font_thickness) 

        
         

        # Update the global tracker for the next frame
        last_tracked_faces = current_tracked_faces 
        (h, w) = frame.shape[:2]
        scale_w = MAX_DISPLAY_WIDTH / w
        scale_h = MAX_DISPLAY_HEIGHT / h
        scale = min(scale_w, scale_h)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
           
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame

        cv2.imshow('Surveillance Feed (Day 3 Final)', display_frame)
        
        # Slow down the frame rate slightly to prevent the display queue from flooding.
        if cv2.waitKey(25) & 0xFF == ord('q'): break 
        # ----------------------------------------------------------------------
    
    # --- MISSING CLEANUP (Must be outside the while loop) ---
    cap.release()
    cv2.destroyAllWindows()

def calculate_iou(boxA, boxB):
    # boxA and boxB are (startX, startY, endX, endY)
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Global list to hold last known Bbox and ID
last_tracked_faces = [] 
MAX_IOU_DISTANCE = 0.5
# src/utilis/video_utils.py - At the very bottom

if __name__ == '__main__':
    
    # 1. Ask the user for the input source
    user_input = input("Enter video file path (or type 'live' for webcam): ").strip()

    if user_input.lower() == 'live':
        # Use the webcam: 0 typically refers to the default camera
        INPUT_SOURCE = 0 
        print("Starting live camera feed...")
        
    elif os.path.exists(user_input):
        # Use the provided file path
        INPUT_SOURCE = user_input
        print(f"Starting video file: {user_input}")
        
    else:
        # Fallback to the hardcoded video path if user input is invalid/empty
        INPUT_SOURCE = 'output_tracked_video (1).mp4' 
        
        if not os.path.exists(INPUT_SOURCE):
            print(f"ERROR: File not found at '{INPUT_SOURCE}'. Please verify path or provide a valid input.")
            sys.exit(1) # Exit the program if no valid source is found

    # 2. Run the processing function with the chosen source
    process_video_feed(INPUT_SOURCE)

