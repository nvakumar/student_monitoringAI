import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from tensorflow.keras.models import load_model
from queue import Queue, Full, Empty # Import Queue for thread-safe communication

# ===============================
# Initialization
# ===============================

# Load the emotion detection model.
# A try-except block is added for robust error handling if the model file is not found.
try:
    model = load_model("models/best_mobilenet_model.h5")
    print("✅ Emotion model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}. Please ensure 'models/best_mobilenet_model.h5' exists in the correct directory.")
    # In a production environment, you might want to exit or provide a fallback.
    # For now, we'll let the program continue, but emotion detection won't work.
    model = None # Set model to None if loading fails

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MediaPipe Pose and Face Mesh solutions.
# These are used for posture, attention, and eye status detection.
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Haar Cascade classifier for face detection.
# This is specifically used to crop faces for the emotion detection model.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define thresholds for various detections
EYE_AR_THRESH = 0.25        # Eye Aspect Ratio threshold for eye closure
INA_ATTENTION_THRESHOLD = 0.20 # Threshold for detecting inattention based on nose position
POSTURE_THRESHOLD = 0.08    # Threshold for detecting slouching posture
IMG_SIZE = 160              # Input image size for the emotion detection model

# NEW: Define a scale factor for processing frames.
# Frames will be resized to this factor before being sent to the processing thread.
# This significantly reduces the computational load for MediaPipe and the emotion model.
PROCESSING_SCALE_FACTOR = 0.5 # Example: 0.5 means half resolution (e.g., 640x480 becomes 320x240)

# Thread-safe shared dictionary to store the latest detected status.
# This dictionary is updated by the processing thread and read by the main thread.
lock = threading.Lock() # Lock to ensure thread-safe access to detected_status
detected_status = {
    "detected_student_id": "Unknown",
    "emotion": "Neutral",
    "posture": "Good",
    "eyes_status": "Open",
    "attention": "Focused",
    "confidence": 0.0,
    "face_bbox": (0, 0, 0, 0), # Added to store face bounding box (x, y, w, h)
    "last_updated": time.time()
}

# Queue for passing frames from the main video capture thread to the background processing thread.
# A small maxsize helps prevent the main thread from blocking if processing lags.
frame_queue = Queue(maxsize=2)

# ===============================
# Helper Functions
# ===============================

def eye_aspect_ratio(landmarks, left=True):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given eye using MediaPipe landmarks.
    EAR is used to determine if eyes are open or closed.
    
    Args:
        landmarks: MediaPipe face_mesh landmarks object.
        left (bool): True for left eye, False for right eye.
        
    Returns:
        float: The calculated Eye Aspect Ratio.
    """
    # Define indices for the eye landmarks based on MediaPipe's Face Mesh model.
    # These points form the shape of the eye.
    # Left eye: horizontal (33, 133), vertical (160, 158, 153, 144)
    # Right eye: horizontal (362, 263), vertical (385, 387, 373, 380)
    points = [33, 160, 158, 133, 153, 144] if left else [362, 385, 387, 263, 373, 380]
    
    # Extract (x, y) coordinates for the specified eye landmarks.
    # Note: MediaPipe landmarks are normalized (0 to 1).
    p = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in points]
    
    # Compute the Euclidean distances between the two sets of vertical eye landmarks.
    # These represent the height of the eye opening.
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5])) # Distance between vertical points 1 and 5
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4])) # Distance between vertical points 2 and 4
    
    # Compute the Euclidean distance between the horizontal eye landmark.
    # This represents the width of the eye.
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3])) # Distance between horizontal points 0 and 3
    
    # Calculate the Eye Aspect Ratio (EAR) using the formula: (A + B) / (2.0 * C)
    ear = (A + B) / (2.0 * C)
    return ear

# ===============================
# Background Processing Thread
# ===============================

def process_frames():
    """
    This function runs in a separate thread and performs all the computationally
    intensive tasks like MediaPipe pose and face mesh processing, and emotion prediction.
    It continuously fetches frames from `frame_queue` and updates `detected_status`.
    """
    emotion_buffer = [] # Buffer to store recent emotion predictions for smoothing
    buffer_size = 5     # Number of frames to average emotion over

    while True:
        try:
            # Attempt to get a frame from the queue.
            # `timeout=1` prevents indefinite blocking if the main thread stops pushing frames.
            # `original_h, original_w` are passed to scale back bounding box coordinates.
            frame_rgb, frame_gray, original_h, original_w = frame_queue.get(timeout=1)
        except Empty:
            # If the queue is empty, continue looping to try again.
            continue

        # Initialize local variables for this frame's detection results.
        current_posture = "Good"
        current_eyes_status = "Open"
        current_attention = "Focused"
        current_emotion = "Neutral"
        current_confidence = 0.0
        current_face_bbox = (0, 0, 0, 0) # Default empty bounding box

        # === Posture Detection (MediaPipe Pose) ===
        # Process the RGB frame to detect body pose landmarks.
        pose_result = pose.process(frame_rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            # Check if necessary landmarks (nose, shoulders) are detected before calculation.
            if len(lm) > 12: 
                # Simple posture check: If the nose (lm[0]) is significantly below the
                # midpoint of the shoulders (lm[11] and lm[12]), it indicates slouching.
                if lm[0].y - (lm[11].y + lm[12].y) / 2 > POSTURE_THRESHOLD:
                    current_posture = "Slouching"

        # === Face Landmarks: Attention + Eyes (MediaPipe Face Mesh) ===
        # Process the RGB frame to detect face mesh landmarks.
        face_result = face_mesh.process(frame_rgb)
        if face_result.multi_face_landmarks:
            face = face_result.multi_face_landmarks[0] # Get the first detected face

            # Attention detection: Check if the nose (landmark 1) is too far off-center horizontally.
            if face.landmark[1]: # Ensure nose landmark exists
                nose = face.landmark[1]
                # If the absolute difference from the center (0.5) exceeds the threshold, it's inattentive.
                if abs(nose.x - 0.5) > INA_ATTENTION_THRESHOLD:
                    current_attention = "Inattentive"

            # Eye status detection: Calculate Eye Aspect Ratio (EAR) for both eyes.
            # Check if all required eye landmarks are present before calculating EAR.
            if len(face.landmark) > 387: 
                ear_l = eye_aspect_ratio(face, True)  # EAR for left eye
                ear_r = eye_aspect_ratio(face, False) # EAR for right eye
                
                # If the average EAR falls below the threshold, eyes are considered closed.
                if (ear_l + ear_r) / 2 < EYE_AR_THRESH:
                    current_eyes_status = "Closed"
            
            # Optionally, draw face mesh landmarks on the frame if needed for debugging/visualization
            # This part is commented out as drawing is handled in the main thread for overlay.
            # mp_draw.draw_landmarks(frame_rgb, face, mp_face.FACEMESH_TESSELATION,
            #                        landmark_drawing_spec=None,
            #                        connection_drawing_spec=mp_draw.DrawingSpec(color=(100, 100, 100), thickness=1))

        # === Emotion Detection (Haar Cascade + Keras Model) ===
        # Detect faces using the Haar Cascade classifier on the grayscale frame.
        # This is typically faster than MediaPipe for just bounding box detection.
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        
        # Process only the first detected face for emotion prediction.
        if len(faces) > 0 and model is not None: # Ensure a face is detected and model is loaded
            (x, y, w_face, h_face) = faces[0] # Get coordinates of the first detected face
            
            # Ensure the detected face is large enough for reliable emotion prediction.
            if w_face >= 50 and h_face >= 50:
                # Scale the bounding box coordinates back to the original frame size
                scaled_x = int(x / PROCESSING_SCALE_FACTOR)
                scaled_y = int(y / PROCESSING_SCALE_FACTOR)
                scaled_w = int(w_face / PROCESSING_SCALE_FACTOR)
                scaled_h = int(h_face / PROCESSING_SCALE_FACTOR)
                current_face_bbox = (scaled_x, scaled_y, scaled_w, scaled_h) # Store the scaled bounding box
                
                # Extract the face region from the RGB frame (which is the downscaled frame).
                face_img = frame_rgb[y:y + h_face, x:x + w_face]
                
                # Resize the face image to the model's input size and normalize pixel values.
                face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
                
                # Predict emotion using the loaded Keras model. `verbose=0` suppresses output.
                preds = model.predict(np.expand_dims(face_resized, axis=0), verbose=0)
                class_id = np.argmax(preds) # Get the index of the highest prediction score
                
                # Add the predicted emotion and its confidence to the buffer for smoothing.
                emotion_buffer.append((class_names[class_id], preds[0][class_id]))
                if len(emotion_buffer) > buffer_size:
                    emotion_buffer.pop(0) # Remove the oldest entry if the buffer is full.

        # === Smooth Emotion (if buffer has data) ===
        # If there are emotion predictions in the buffer, average them for a smoother output.
        if emotion_buffer:
            emo_scores = {}
            for emo, score in emotion_buffer:
                emo_scores[emo] = emo_scores.get(emo, 0) + score
            
            # Determine the emotion with the highest accumulated score in the buffer.
            current_emotion = max(emo_scores, key=emo_scores.get)
            # Calculate the average confidence for the dominant emotion.
            current_confidence = emo_scores[current_emotion] / len(emotion_buffer)

        # === Update Shared State (Thread-Safe) ===
        # Acquire the lock before updating the shared `detected_status` dictionary
        # to prevent race conditions with the main thread.
        with lock:
            detected_status.update({
                "emotion": current_emotion,
                "posture": current_posture,
                "eyes_status": current_eyes_status,
                "attention": current_attention,
                "confidence": current_confidence,
                "face_bbox": current_face_bbox, # Update the bounding box in shared state
                "last_updated": time.time()
            })

# Start the background processing thread when the script runs.
# `daemon=True` ensures the thread will automatically terminate when the main program exits.
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()
print("✅ Background processing thread started.")

# ===============================
# Frame Generators for Flask/Web Streaming
# ===============================

def gen_frames_multi():
    """
    Generator function to yield frames from the student's webcam.
    This function primarily handles video capture, overlay drawing, and streaming.
    The heavy processing is offloaded to the `process_frames` thread.
    """
    cap = None
    # Try multiple camera indices to find an available student webcam
    for i in range(5): # Try indices 0, 1, 2, 3, 4
        print(f"Attempting to open student webcam at index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Student webcam opened successfully at index {i}.")
            break
        else:
            print(f"❌ Failed to open student webcam at index {i}.")
            cap.release() # Release if not opened to avoid resource leaks
            time.sleep(0.5) # Small delay before trying next index
    
    if not cap or not cap.isOpened():
        print("❌ All attempts to open student webcam failed. Please ensure it's connected and not in use by another application.")
        return # Exit if webcam cannot be opened

    while True:
        success, frame = cap.read() # Read a frame from the webcam
        if not success:
            print("Failed to read frame from student webcam. Breaking loop.")
            break # Break the loop if frame reading fails

        # Get original frame dimensions
        original_h, original_w, _ = frame.shape

        # Downscale the frame for background processing
        processing_w = int(original_w * PROCESSING_SCALE_FACTOR)
        processing_h = int(original_h * PROCESSING_SCALE_FACTOR)
        processing_frame = cv2.resize(frame, (processing_w, processing_h), interpolation=cv2.INTER_AREA)

        # Convert the downscaled frame to RGB and Grayscale for processing.
        rgb_processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        gray_processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        
        # Try to put the converted, downscaled frames and original dimensions into the queue.
        try:
            frame_queue.put_nowait((rgb_processing_frame, gray_processing_frame, original_h, original_w))
        except Full:
            # print("Queue full, dropping frame for smoother video.") # Uncomment for debugging
            pass

        # === Retrieve Latest Status for Display (Thread-Safe) ===
        # Acquire the lock to safely read the `detected_status` dictionary.
        # A copy is made to release the lock quickly, allowing the processing thread to update.
        current_status = {}
        with lock:
            current_status = detected_status.copy()

        # === Draw Overlays on the original frame ===

        # 1. Draw bounding box around the detected face for emotion.
        bbox = current_status.get("face_bbox", (0, 0, 0, 0))
        # Only draw if a valid bounding box (width and height > 0) is available.
        # The bbox coordinates are already scaled back to original frame size by the processing thread.
        if bbox[2] > 0 and bbox[3] > 0: 
            x, y, w, h = bbox
            # Draw a cyan rectangle around the face.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) 

        # 2. Display detected emotion and its confidence.
        # Text color: Yellow (255, 255, 0)
        cv2.putText(frame, f"Emotion: {current_status['emotion']} ({current_status['confidence']:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) 

        # 3. Display posture, attention, and eye status.
        # Text color: Green (0, 255, 0)
        cv2.putText(frame, f"Posture: {current_status['posture']} | Attention: {current_status['attention']} | Eyes: {current_status['eyes_status']}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode the processed frame to JPEG format.
        # This is necessary for streaming video over HTTP (e.g., to a Flask app).
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in multipart/x-mixed-replace format.
        # This is the standard format for streaming MJPEG video in web applications.
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release() # Release the webcam resource when the loop finishes.
    print("Student webcam released.")


def get_latest_status():
    """
    Returns a copy of the current detected status.
    This function can be called by other parts of the application (e.g., a dashboard)
    to retrieve the latest analytical results.
    """
    with lock: # Ensure thread-safe access
        return detected_status.copy()


def gen_frames_teacher():
    """
    Generator function to yield frames from the teacher's webcam.
    This function operates independently and is not involved in the student monitoring logic.
    """
    cap = None
    # Try multiple camera indices to find an available teacher webcam
    for i in range(5): # Try indices 0, 1, 2, 3, 4
        print(f"Attempting to open teacher webcam at index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Teacher webcam opened successfully at index {i}.")
            break
        else:
            print(f"❌ Failed to open teacher webcam at index {i}.")
            cap.release() # Release if not opened to avoid resource leaks
            time.sleep(0.5) # Small delay before trying next index

    if not cap or not cap.isOpened():
        print("❌ All attempts to open teacher webcam failed. Please ensure it's connected and not in use.")
        return
        
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from teacher webcam. Breaking loop.")
            break
        
        # Add a text overlay to clearly identify this as the teacher's camera.
        cv2.putText(frame, "Teacher Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Cyan text
        
        # Encode the frame to JPEG and yield it for streaming.
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release() # Release the webcam resource.
    print("Teacher webcam released.")
