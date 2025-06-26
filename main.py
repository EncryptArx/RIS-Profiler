import cv2
import numpy as np
from face_utils import setup_face_app, load_known_faces
from utils.logging_utils import log_detection
import mediapipe as mp
from ultralytics import YOLO
from collections import defaultdict, deque
import playsound
import threading
import logging
from utils.liveness_utils import setup_liveness, is_blinking
import time # Import time module

# MediaPipe setup for drawing utilities and Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Path to the alert sound file
ALERT_SOUND_PATH = "alerts/alert.mp3"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture DEBUG level messages and higher
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log", mode='w'),  # Log to a file, overwrite each run
        logging.StreamHandler()  # Log to console
    ]
)

def play_alert_sound():
    """Plays the alert sound in a separate thread."""
    try:
        logging.debug("Attempting to play alert sound.")
        playsound.playsound(ALERT_SOUND_PATH, block=False)
        logging.debug("Alert sound played successfully.") # Uncomment for debugging sound playback
    except Exception as e:
        logging.error(f"Could not play sound: {e}")

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
    print("[INFO] Successfully loaded YOLOv8 model for person detection.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLOv8 model: {e}")
    print("[ERROR] Make sure you have installed ultralytics package and have internet connection to download the model.")
    model = None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class IDManager:
    def __init__(self):
        self.next_id = 1
        self.active_ids = {}  # {yolo_id: our_id}
        self.id_history = {}  # {our_id: {last_seen, embedding, name, confidence, triggered_alert, position_history, loitering_alert_triggered, running_alert_triggered, intrusion_alert_triggered, pose_history}}
        self.available_ids = set()
        self.known_face_ids = {}  # {name: our_id}
        self.loitering_threshold = 10 # seconds
        self.loitering_movement_threshold = 30 # pixels
        self.running_speed_threshold = 50 # pixels per second (adjust as needed)
        self.position_history_window = 30 # frames (~1 second at 30 FPS) for speed calculation
        self.pose_history_window = 60 # frames for pose analysis
        self.fighting_proximity_threshold = 150 # pixels (distance between centers)
        self.fighting_movement_threshold = 20 # pixels (cumulative joint movement)
        self.kicking_speed_threshold = 2 # pixels per frame (lowered further for testing)
        self.punching_speed_threshold = 5 # pixels per frame (lowered for testing)
        self.falling_speed_threshold = 10 # pixels per frame (lowered for testing)
        self.recognition_alert_timeout = 5  # seconds to reset recognition alert

    def get_id(self, yolo_id, embedding=None, name="Unknown"):
        # If we have a known face, try to maintain its ID
        if name != "Unknown":
            if name in self.known_face_ids:
                our_id = self.known_face_ids[name]
                self.active_ids[yolo_id] = our_id
                self.id_history[our_id]['last_seen'] = 0
                self.id_history[our_id]['embedding'] = embedding
                self.id_history[our_id]['name'] = name
                self.id_history[our_id]['confidence'] = 1.0
                self.id_history[our_id]['triggered_alert'] = False
                self.id_history[our_id]['loitering_alert_triggered'] = False
                self.id_history[our_id]['running_alert_triggered'] = False
                self.id_history[our_id]['intrusion_alert_triggered'] = False
                self.id_history[our_id]['falling_alert_triggered'] = False
                self.id_history[our_id]['fighting_alert_triggered'] = False
                self.id_history[our_id]['kicking_alert_triggered'] = False
                self.id_history[our_id]['punching_alert_triggered'] = False
                # Initialize position history if it doesn't exist (for re-identified known faces)
                if 'position_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['position_history'] = deque(maxlen=60)
                # Initialize pose history if it doesn't exist
                if 'pose_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['pose_history'] = deque(maxlen=self.pose_history_window)
                return our_id
            else:
                # New known face, assign next available ID
                if self.available_ids:
                    our_id = min(self.available_ids)
                    self.available_ids.remove(our_id)
                else:
                    our_id = self.next_id
                    self.next_id += 1
                
                self.known_face_ids[name] = our_id
                self.active_ids[yolo_id] = our_id
                self.id_history[our_id] = {
                    'last_seen': 0,
                    'embedding': embedding,
                    'name': name,
                    'confidence': 1.0,
                    'triggered_alert': False,
                    'recognition_alert_logged': False,
                    'recognition_alert_time': 0,
                    'position_history': deque(maxlen=60),
                    'loitering_alert_triggered': False,
                    'running_alert_triggered': False,
                    'intrusion_alert_triggered': False,
                    'falling_alert_triggered': False,
                    'fighting_alert_triggered': False,
                    'kicking_alert_triggered': False,
                    'punching_alert_triggered': False,
                    'pose_history': deque(maxlen=self.pose_history_window)
                }
                return our_id

        # For unknown faces, try to re-identify using embedding
        if embedding is not None:
            best_match = None
            best_similarity = 0.6  # Minimum similarity threshold

            # First try to match with known faces
            for known_id, data in self.id_history.items():
                if data['embedding'] is not None and data['name'] != "Unknown":
                    similarity = cosine_similarity(embedding, data['embedding'])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (known_id, data['name'])

            if best_match is not None:
                our_id, matched_name = best_match
                self.active_ids[yolo_id] = our_id
                self.id_history[our_id]['last_seen'] = 0
                self.id_history[our_id]['embedding'] = embedding
                self.id_history[our_id]['name'] = matched_name
                self.known_face_ids[matched_name] = our_id
                # Initialize position history if it doesn't exist (for re-identified known faces)
                if 'position_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['position_history'] = deque(maxlen=60)
                # Also initialize alert flags if they don't exist
                if 'loitering_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['loitering_alert_triggered'] = False
                if 'running_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['running_alert_triggered'] = False
                if 'intrusion_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['intrusion_alert_triggered'] = False
                if 'falling_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['falling_alert_triggered'] = False
                if 'fighting_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['fighting_alert_triggered'] = False
                if 'kicking_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['kicking_alert_triggered'] = False
                if 'punching_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['punching_alert_triggered'] = False
                # Initialize pose history if it doesn't exist
                if 'pose_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['pose_history'] = deque(maxlen=self.pose_history_window)
                return our_id

            # If no match with known faces, try to match with unknown faces
            for known_id, data in self.id_history.items():
                if data['embedding'] is not None and data['name'] == "Unknown":
                    similarity = cosine_similarity(embedding, data['embedding'])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (known_id, "Unknown")

            if best_match is not None:
                our_id, _ = best_match
                self.active_ids[yolo_id] = our_id
                self.id_history[our_id]['last_seen'] = 0
                self.id_history[our_id]['embedding'] = embedding
                # Initialize position history if it doesn't exist (for re-identified unknown faces)
                if 'position_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['position_history'] = deque(maxlen=60)
                # Also initialize alert flags if they don't exist
                if 'loitering_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['loitering_alert_triggered'] = False
                if 'running_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['running_alert_triggered'] = False
                if 'intrusion_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['intrusion_alert_triggered'] = False
                if 'falling_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['falling_alert_triggered'] = False
                if 'fighting_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['fighting_alert_triggered'] = False
                if 'kicking_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['kicking_alert_triggered'] = False
                if 'punching_alert_triggered' not in self.id_history[our_id]:
                    self.id_history[our_id]['punching_alert_triggered'] = False
                # Initialize pose history if it doesn't exist
                if 'pose_history' not in self.id_history[our_id]:
                    self.id_history[our_id]['pose_history'] = deque(maxlen=self.pose_history_window)
                return our_id

        # If no match found, assign new ID
        if self.available_ids:
            our_id = min(self.available_ids)
            self.available_ids.remove(our_id)
        else:
            our_id = self.next_id
            self.next_id += 1

        self.active_ids[yolo_id] = our_id
        self.id_history[our_id] = {
            'last_seen': 0,
            'embedding': embedding,
            'name': "Unknown",
            'confidence': 0.0,
            'triggered_alert': False,
            'recognition_alert_logged': False,
            'recognition_alert_time': 0,
            'position_history': deque(maxlen=60),
            'loitering_alert_triggered': False,
            'running_alert_triggered': False,
            'intrusion_alert_triggered': False,
            'falling_alert_triggered': False,
            'fighting_alert_triggered': False,
            'kicking_alert_triggered': False,
            'punching_alert_triggered': False,
            'pose_history': deque(maxlen=self.pose_history_window)
        }
        return our_id

    def update(self):
        # Update last_seen for all active IDs
        for our_id, data in self.id_history.items():
            data['last_seen'] += 1
            # Reset recognition_alert_logged if not seen for a while
            if data.get('recognition_alert_logged', False):
                # If not seen for more than recognition_alert_timeout seconds, reset
                if len(data['position_history']) > 0:
                    last_time = data['position_history'][-1][0]
                    if time.time() - last_time > self.recognition_alert_timeout:
                        data['recognition_alert_logged'] = False
            # Reset triggered_alert for IDs that haven't been seen recently
            if data['last_seen'] > 30:  # Use the same threshold as for removal
                data['triggered_alert'] = False
                # Also reset behavior-specific alert flags
                data['loitering_alert_triggered'] = False
                data['running_alert_triggered'] = False
                data['intrusion_alert_triggered'] = False
                data['falling_alert_triggered'] = False
                data['fighting_alert_triggered'] = False
                data['kicking_alert_triggered'] = False
                data['punching_alert_triggered'] = False

        # Remove inactive IDs, but preserve known face IDs
        for yolo_id, our_id in list(self.active_ids.items()):
            if self.id_history[our_id]['last_seen'] > 30:  # 30 frames = ~1 second
                del self.active_ids[yolo_id]
                # Only add to available_ids if it's not a known face
                if self.id_history[our_id]['name'] == "Unknown":
                    self.available_ids.add(our_id)
                # Don't delete from id_history to maintain known face information

    def calculate_speed(self, our_id):
        """Calculates the speed of a tracked object based on position history."""
        history = self.id_history[our_id]['position_history']
        if len(history) < 2 or len(history) < self.position_history_window:
            return 0.0 # Not enough history to calculate speed

        # Get positions from the defined window
        start_time, start_pos = history[-self.position_history_window]
        end_time, end_pos = history[-1]

        duration = end_time - start_time
        if duration == 0:
            return 0.0 # Avoid division by zero

        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))

        speed = distance / duration # pixels per second
        return speed

# Define restricted zones as lists of points (polygons)
# Example: A single rectangular zone covering the door area on the left
RESTRICTED_ZONES = [
    np.array([[50, 0], [450, 0], [450, 720], [50, 720]], np.int32)
]

def is_point_in_polygon(point, polygon):
    """Checks if a point is inside a polygon."""
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def calculate_depth(box_width, box_height, frame_width, frame_height):
    """Calculate relative depth based on bounding box size"""
    # Normalize the box size relative to frame size
    normalized_size = (box_width * box_height) / (frame_width * frame_height)
    # Convert to a depth value (smaller box = further away)
    depth = 1.0 / normalized_size
    return depth

def is_falling_pose(pose_history, frame_height, id_manager):
    """Analyzes pose history to detect if a person is falling."""
    logging.debug(f"Checking for falling pose. History length: {len(pose_history)}") # Added debug log
    if len(pose_history) < 10: # Need enough history to determine falling
        logging.debug("Not enough pose history for falling detection.") # Added debug log
        return False

    # Get hip joint positions over the last few frames
    # We'll use the average of left and right hip for a more robust measure
    hip_indices = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    recent_hip_y = []

    for _, pose_landmarks in list(pose_history)[-10:]:
        if pose_landmarks:
            visible_hips_y = [pose_landmarks[i][1] for i in hip_indices if pose_landmarks[i] is not None and pose_landmarks[i][3] > 0.5] # Access y using index [1]
            if visible_hips_y:
                recent_hip_y.append(np.mean(visible_hips_y) * frame_height)

    if len(recent_hip_y) < 10: # Not enough valid hip positions in history
        logging.debug(f"Not enough valid hip positions ({len(recent_hip_y)}) in history for falling detection.") # Added debug log
        return False

    # Calculate the change in vertical position (Y-coordinate) of the hips
    # A significant downward movement indicates a potential fall
    vertical_change = recent_hip_y[-1] - recent_hip_y[0]
    logging.debug(f"Falling detection: Vertical change in hip position over last 10 frames: {vertical_change:.2f} px. Threshold: {id_manager.falling_speed_threshold}") # Added debug log

    # Check if the vertical change exceeds the falling speed threshold (a negative value for downward movement)
    # We are looking for a significant *negative* change in y-coordinate (downward movement)
    if vertical_change < -id_manager.falling_speed_threshold: # Falling is downward movement, so expect a negative change
        logging.debug(f"Falling detected! Vertical change: {vertical_change:.2f} px < -Threshold: {-id_manager.falling_speed_threshold}") # Added debug log
        return True

    return False

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at point b given three points a, b, c (as (x, y))."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_fighting_pose(pose_history1, pose_history2, frame_width, frame_height, id_manager):
    """Analyzes pose histories of two people to detect fighting."""
    if len(pose_history1) < 5 or len(pose_history2) < 5: # Need some history
        return False

    # Get recent pose landmarks for both individuals
    recent_poses1 = list(pose_history1)[-5:]
    recent_poses2 = list(pose_history2)[-5:]

    # Check for close proximity based on a central point (e.g., average of hips and shoulders)
    def get_center_point(pose_landmarks, w, h):
        if not pose_landmarks:
            return None
        relevant_landmarks = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        points = [(pose_landmarks[i][0] * w, pose_landmarks[i][1] * h) for i in relevant_landmarks if pose_landmarks[i] is not None and pose_landmarks[i][3] > 0.5]
        if not points:
            return None
        return np.mean(points, axis=0)

    center1 = get_center_point(recent_poses1[-1][1], frame_width, frame_height)
    center2 = get_center_point(recent_poses2[-1][1], frame_width, frame_height)

    if center1 is None or center2 is None:
        return False

    distance = np.linalg.norm(np.array(center1) - np.array(center2))

    # Check if they are close enough
    if distance > id_manager.fighting_proximity_threshold: # Use the threshold from IDManager
        return False

    # Analyze movement of upper body joints over the recent frames
    def calculate_movement(pose_history, w, h):
        if len(pose_history) < 2:
            return 0

        relevant_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]

        total_movement = 0
        for i in range(1, len(pose_history)):
            _, pose1 = pose_history[i-1]
            _, pose2 = pose_history[i]
            if pose1 and pose2:
                for landmark_idx in relevant_landmarks:
                    lm1 = pose1[landmark_idx]
                    lm2 = pose2[landmark_idx]
                    if lm1 is not None and lm2 is not None and lm1[3] > 0.5 and lm2[3] > 0.5:
                        p1 = np.array([lm1[0] * w, lm1[1] * h])
                        p2 = np.array([lm2[0] * w, lm2[1] * h])
                        total_movement += np.linalg.norm(p2 - p1)
        return total_movement

    movement1 = calculate_movement(recent_poses1, frame_width, frame_height)
    movement2 = calculate_movement(recent_poses2, frame_width, frame_height)

    # Check if both individuals have significant upper body movement
    if movement1 > id_manager.fighting_movement_threshold and movement2 > id_manager.fighting_movement_threshold: # Use threshold from id_manager
        logging.debug(f"Potential fighting detected. Distance: {distance:.2f}px, Movement1: {movement1:.2f}px, Movement2: {movement2:.2f}px")
        return True

    return False

def is_kicking_pose(pose_history, frame_width, frame_height, id_manager):
    """Analyzes pose history to detect if a person is kicking."""
    # logging.debug(f"Checking for kicking pose. History length: {len(pose_history)}") # Added debug log
    if len(pose_history) < 5: # Need enough history
        # logging.debug("Not enough pose history for kicking detection.") # Added debug log
        return False

    # Check for rapid movement of ankle relative to hip over the last few frames
    relevant_landmarks = [
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
    ]

    recent_poses = list(pose_history)[-5:]

    for i in range(1, len(recent_poses)):
        timestamp1, pose1 = recent_poses[i-1]
        timestamp2, pose2 = recent_poses[i]

        if pose1 and pose2:
            for leg_side in [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]:
                hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value if leg_side == mp_pose.PoseLandmark.LEFT_ANKLE.value else mp_pose.PoseLandmark.RIGHT_HIP.value

                lm1_ankle = pose1[leg_side]
                lm1_hip = pose1[hip_idx]
                lm2_ankle = pose2[leg_side]
                lm2_hip = pose2[hip_idx]

                # Ensure landmarks are visible and have sufficient confidence
                if lm1_ankle and lm1_hip and lm2_ankle and lm2_hip and \
                   lm1_ankle[3] > 0.5 and lm1_hip[3] > 0.5 and \
                   lm2_ankle[3] > 0.5 and lm2_hip[3] > 0.5:

                    # Calculate movement of ankle relative to hip
                    # Using relative movement helps account for overall body movement
                    delta_ankle_x = (lm2_ankle[0] - lm1_ankle[0]) * frame_width
                    delta_ankle_y = (lm2_ankle[1] - lm1_ankle[1]) * frame_height
                    delta_hip_x = (lm2_hip[0] - lm1_hip[0]) * frame_width
                    delta_hip_y = (lm2_hip[1] - lm1_hip[1]) * frame_height

                    relative_movement_x = abs(delta_ankle_x - delta_hip_x)
                    relative_movement_y = abs(delta_ankle_y - delta_hip_y)

                    # Calculate speed as distance moved per frame
                    movement_speed = (relative_movement_x**2 + relative_movement_y**2)**0.5 # euclidean distance
                    # logging.debug(f"Kicking detection: Leg side {leg_side}, Movement speed: {movement_speed}") # Added debug log

                    if movement_speed > id_manager.kicking_speed_threshold:
                        # logging.debug(f"Kicking detected! Speed: {movement_speed} > Threshold: {id_manager.kicking_speed_threshold}") # Added debug log
                        return True

    return False

def is_punching_pose(pose_history, frame_width, frame_height, id_manager, others_positions=None):
    """Analyzes pose history to detect if a person is punching, with elbow angle, direction, speed, and nearby person checks."""
    if len(pose_history) < 5:
        return False

    recent_poses = list(pose_history)[-5:]
    punch_detected = False

    for i in range(1, len(recent_poses)):
        _, pose1 = recent_poses[i-1]
        _, pose2 = recent_poses[i]
        if pose1 and pose2:
            for arm_side in [mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]:
                shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value if arm_side == mp_pose.PoseLandmark.LEFT_WRIST.value else mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value if arm_side == mp_pose.PoseLandmark.LEFT_WRIST.value else mp_pose.PoseLandmark.RIGHT_ELBOW.value

                lm1_wrist = pose1[arm_side]
                lm2_wrist = pose2[arm_side]
                lm1_elbow = pose1[elbow_idx]
                lm2_elbow = pose2[elbow_idx]
                lm1_shoulder = pose1[shoulder_idx]
                lm2_shoulder = pose2[shoulder_idx]

                if (lm1_wrist and lm2_wrist and lm1_elbow and lm2_elbow and lm1_shoulder and lm2_shoulder and
                    lm1_wrist[3] > 0.5 and lm2_wrist[3] > 0.5 and
                    lm1_elbow[3] > 0.5 and lm2_elbow[3] > 0.5 and
                    lm1_shoulder[3] > 0.5 and lm2_shoulder[3] > 0.5):

                    # Convert normalized to pixel coordinates
                    w1 = np.array([lm1_wrist[0] * frame_width, lm1_wrist[1] * frame_height])
                    w2 = np.array([lm2_wrist[0] * frame_width, lm2_wrist[1] * frame_height])
                    e1 = np.array([lm1_elbow[0] * frame_width, lm1_elbow[1] * frame_height])
                    e2 = np.array([lm2_elbow[0] * frame_width, lm2_elbow[1] * frame_height])
                    s1 = np.array([lm1_shoulder[0] * frame_width, lm1_shoulder[1] * frame_height])
                    s2 = np.array([lm2_shoulder[0] * frame_width, lm2_shoulder[1] * frame_height])

                    # 1. Elbow angle check
                    angle1 = calculate_angle(w1, e1, s1)
                    angle2 = calculate_angle(w2, e2, s2)
                    if not (angle1 < 60 and angle2 > 150):
                        continue  # Only trigger if elbow goes from bent to straight

                    # 2. Direction check: wrist should move away from shoulder
                    move_vec = w2 - w1
                    shoulder_to_wrist_vec = w1 - s1
                    if np.dot(move_vec, shoulder_to_wrist_vec) < 0:
                        continue  # Not moving away from shoulder
                    # Check if movement is roughly in the same direction as shoulder-to-wrist
                    move_dir = move_vec / (np.linalg.norm(move_vec) + 1e-6)
                    sw_dir = shoulder_to_wrist_vec / (np.linalg.norm(shoulder_to_wrist_vec) + 1e-6)
                    if np.dot(move_dir, sw_dir) < 0.7:
                        continue  # Not a straight punch

                    # 3. Minimum speed/distance
                    wrist_movement = np.linalg.norm(move_vec)
                    if wrist_movement < max(30, id_manager.punching_speed_threshold):
                        continue  # Not fast enough

                    # 4. (Optional) Check for nearby person in punch direction
                    if others_positions:
                        punch_tip = w2
                        punch_dir = move_dir
                        for other_pos in others_positions:
                            # Project vector from shoulder to other person onto punch direction
                            to_other = np.array(other_pos) - s2
                            proj = np.dot(to_other, punch_dir)
                            if 0 < proj < 200:  # within 200px in punch direction
                                # Perpendicular distance from punch line
                                perp_dist = np.linalg.norm(to_other - proj * punch_dir)
                                if perp_dist < 100:  # within 100px of punch line
                                    punch_detected = True
                                    break
                        if punch_detected:
                            logging.debug(f"Punching detected with nearby person in direction. Angle1: {angle1:.1f}, Angle2: {angle2:.1f}, Wrist move: {wrist_movement:.1f}")
                            return True
                    else:
                        punch_detected = True
                        logging.debug(f"Punching detected. Angle1: {angle1:.1f}, Angle2: {angle2:.1f}, Wrist move: {wrist_movement:.1f}")
                        return True
    return False

def main():
    app = setup_face_app()
    logging.info("Loading known faces...")
    known_encodings, known_names = load_known_faces(app)

    if not known_encodings:
        logging.error("No known faces found.")
        return

    # For CCTV, you can use either webcam (0) or video file
    # cap = cv2.VideoCapture(0)  # For webcam
    # cap = cv2.VideoCapture("path_to_cctv_feed.mp4")  # For video file
    cap = cv2.VideoCapture(0)  # Currently using webcam for testing
    
    # Set video properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    
    logging.info("Video feed started. Press 'q' to quit.")
    logging.info("Press 'd' to toggle depth visualization.")

    frame_count = 0
    show_depth = False
    process_every_n_frames = 2  # Process every 2nd frame for better performance
    start_time = time.time() # Get start time for calculating duration

    # Parameters for face recognition
    RECOGNITION_THRESHOLD = 0.6  # Threshold for face recognition similarity
    RECOGNITION_CACHE_TIMEOUT = 10  # seconds
    
    # Initialize ID manager
    id_manager = IDManager()
    
    # Setup liveness detection
    setup_liveness()

    # Setup MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Skip frames for better performance
            if frame_count % process_every_n_frames != 0:
                continue

            h, w = frame.shape[:2]

            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False # Improve performance

            # Inside the main loop, before processing each frame:
            frame_start_time = time.time()

            # --- YOLO Detection ---
            yolo_start = time.time()
            results = model.track(
                frame, 
                persist=True,
                classes=[0],  # class 0 is person in COCO dataset
                tracker="trackers/bytetrack.yaml",  # Use ByteTrack configuration
                conf=0.3,  # Lower confidence threshold for better tracking
                iou=0.5,  # IOU threshold for tracking
                max_det=300  # Maximum number of detections to track
            )
            yolo_end = time.time()
            print(f"YOLO detection time: {(yolo_end - yolo_start) * 1000:.2f} ms")

            # --- Pose Estimation ---
            pose_start = time.time()
            pose_results = pose.process(frame_rgb)
            pose_end = time.time()
            print(f"Pose estimation time: {(pose_end - pose_start) * 1000:.2f} ms")

            # --- Face Recognition & Liveness (per person) ---
            # (Insert timing around recognition/liveness code for each detected person)
            recog_start = time.time()
            # ... face recognition and liveness code ...
            recog_end = time.time()
            print(f"Face recognition & liveness time: {(recog_end - recog_start) * 1000:.2f} ms")

            # --- Behavior Analysis ---
            behavior_start = time.time()
            # ... behavior analysis code (loitering, running, intrusion, falling, fighting, etc.) ...
            behavior_end = time.time()
            print(f"Behavior analysis time: {(behavior_end - behavior_start) * 1000:.2f} ms")

            # --- Total Frame Time ---
            frame_end_time = time.time()
            print(f"Total frame processing time: {(frame_end_time - frame_start_time) * 1000:.2f} ms")

            # Process detections
            if results[0].boxes.id is not None:  # Check if tracking is working
                boxes = results[0].boxes
                
                # Sort boxes by x-coordinate for left-to-right processing
                sorted_indices = np.argsort([box.xyxy[0][0] for box in boxes])
                
                for idx in sorted_indices:
                    box = boxes[idx]
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    yolo_id = int(box.id[0])  # Get the YOLO tracking ID
                    
                    if confidence > 0.45:  # Confidence threshold for person detection
                        # Validate and adjust bounding box coordinates
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        # Calculate box dimensions
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height

                        # Only process if the box is valid and has reasonable size
                        if (x2 > x1 and y2 > y1 and
                            box_area > 400 and
                            box_width < w * 0.95 and
                            box_height < h * 0.95 and
                            box_width > 50 and
                            box_height > 100):
                            
                            # Calculate depth
                            depth = calculate_depth(box_width, box_height, w, h)
                            
                            # Extract person region with padding for better face detection
                            padding = 20
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding)
                            x2_pad = min(w, x2 + padding)
                            y2_pad = min(h, y2 + padding)
                            person_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            
                            # Detect faces within person region
                            faces = app.get(person_region)
                            recognized = False
                            detected_name = "Unknown"
                            current_embedding = None

                            if faces:
                                # Get the first detected face within the person region
                                face = faces[0]
                                current_embedding = face.embedding

                                # Face recognition
                                for idx, known_embedding in enumerate(known_encodings):
                                    similarity = cosine_similarity(current_embedding, known_embedding)
                                    if similarity > RECOGNITION_THRESHOLD:
                                        detected_name = known_names[idx]
                                        recognized = True
                                        break

                            # Perform liveness check
                            is_live = is_blinking(person_region) # Using blink detection as liveness check

                            # Get or assign ID
                            our_id = id_manager.get_id(yolo_id, current_embedding, detected_name)
                            
                            # Calculate center of bounding box and add to position history
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            current_time = time.time()
                            center_point = (center_x, center_y)
                            id_manager.id_history[our_id]['position_history'].append((current_time, center_point))

                            # Store pose landmarks if available
                            if pose_results.pose_landmarks:
                                # Find the pose landmarks corresponding to this person's bounding box
                                person_pose_landmarks = []
                                for landmark in pose_results.pose_landmarks.landmark:
                                    # Check if the landmark is within the person's bounding box
                                    # Note: Landmark coordinates are normalized [0, 1]
                                    if (landmark.x * w > x1 and landmark.x * w < x2 and
                                        landmark.y * h > y1 and landmark.y * h < y2):
                                        person_pose_landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
                                    else:
                                        person_pose_landmarks.append(None) # Append None if landmark is outside box

                                if person_pose_landmarks: # Only append if we found at least one landmark inside the box
                                    id_manager.id_history[our_id]['pose_history'].append((current_time, person_pose_landmarks))
                                else:
                                     id_manager.id_history[our_id]['pose_history'].append((current_time, None)) # Append None if no landmarks found in box

                            # Trigger alert if a known person is recognized and is live
                            if recognized and is_live and not id_manager.id_history[our_id].get('recognition_alert_logged', False):
                                logging.info(f"Triggering alert for recognized and live person ID: {our_id}")
                                threading.Thread(target=play_alert_sound).start()
                                id_manager.id_history[our_id]['recognition_alert_logged'] = True
                                id_manager.id_history[our_id]['recognition_alert_time'] = time.time()
                                # Log the detection and save the frame
                                liveness_status = "Live" if is_live else "Spoof?"
                                log_detection(detected_name, id_manager.id_history[our_id]['confidence'], liveness_status, frame)

                            # Draw bounding box and label
                            color = (0, 255, 0) if recognized else (0, 255, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Put label text with adjusted position
                            text_y = y1 - 10 if y1 - 10 > 20 else y2 + 25
                            text_y = max(20, text_y)
                            label = f"{detected_name} (ID: {our_id})"
                            
                            # Add liveness status to label
                            liveness_status = "Live" if is_live else "Spoof?"
                            label += f" [{liveness_status}]"
                            
                            # Add depth information if enabled
                            if show_depth:
                                label += f" [Depth: {depth:.1f}]"
                                # Add depth visualization bar
                                bar_length = int(50 * (1.0 / depth))  # Normalize depth to bar length
                                bar_length = min(50, max(5, bar_length))  # Clamp between 5 and 50
                                cv2.rectangle(frame, (x2 + 5, y1), (x2 + 5 + bar_length, y1 + 5), (0, 0, 255), -1)
                            
                            cv2.putText(frame, label, (x1, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update ID manager
            id_manager.update()

            # --- Behavior Classification --- #
            # Check for loitering
            for our_id, data in id_manager.id_history.items():
                if len(data['position_history']) > 1:
                    # Get the first and last position in the history
                    first_time, first_pos = data['position_history'][0]
                    last_time, last_pos = data['position_history'][-1]
                    
                    duration = last_time - first_time
                    distance = np.linalg.norm(np.array(last_pos) - np.array(first_pos))
                    
                    # Check for loitering
                    if duration > id_manager.loitering_threshold and distance < id_manager.loitering_movement_threshold and not data.get('loitering_alert_triggered', False):
                        logging.warning(f"Loitering detected for ID {our_id} Duration: {duration:.2f}s, Distance: {distance:.2f}px")
                        # Trigger alert (using the existing alert sound and log)
                        threading.Thread(target=play_alert_sound).start()
                        # Use a simpler name format for filename to avoid issues
                        log_detection(f"Loitering_ID_{our_id}", 1.0, "N/A", frame) # Log with a generic name and high confidence
                        id_manager.id_history[our_id]['loitering_alert_triggered'] = True # Set a flag to avoid repeated alerts
                    # Reset loitering alert flag if the person moves or disappears
                    elif data.get('loitering_alert_triggered', False):
                        id_manager.id_history[our_id]['loitering_alert_triggered'] = False

            # Check for running
            for our_id, data in id_manager.id_history.items():
                if data['name'] != "Unknown": # Only classify behavior for tracked individuals
                    speed = id_manager.calculate_speed(our_id)
                    
                    # Check for running
                    if speed > id_manager.running_speed_threshold and not data.get('running_alert_triggered', False):
                        logging.warning(f"Running detected for ID {our_id} Speed: {speed:.2f} px/s")
                        # Trigger alert
                        threading.Thread(target=play_alert_sound).start()
                        log_detection(f"Running_ID_{our_id}", 1.0, "N/A", frame)
                        id_manager.id_history[our_id]['running_alert_triggered'] = True
                    # Reset running alert flag if the speed drops
                    elif speed < id_manager.running_speed_threshold and data.get('running_alert_triggered', False):
                        id_manager.id_history[our_id]['running_alert_triggered'] = False

            # Check for intrusion
            for our_id, data in id_manager.id_history.items():
                if data['name'] != "Unknown" and len(data['position_history']) > 0: # Only classify behavior for tracked individuals with history
                    current_position = data['position_history'][-1][1] # Get the most recent position
                    
                    # Check if the current position is inside any restricted zone
                    is_intruding = False
                    for i, zone in enumerate(RESTRICTED_ZONES):
                        if is_point_in_polygon(current_position, zone):
                            logging.warning(f"Intrusion detected for ID {our_id} in Zone {i+1}")
                            is_intruding = True
                            break # No need to check other zones if already intruding

                    # Trigger alert if intruding and alert not already triggered
                    if is_intruding and not data.get('intrusion_alert_triggered', False):
                        # Trigger alert
                        threading.Thread(target=play_alert_sound).start()
                        # Use a simpler name format for filename
                        log_detection(f"Intrusion_ID_{our_id}", 1.0, "N/A", frame)
                        id_manager.id_history[our_id]['intrusion_alert_triggered'] = True # Set a flag
                    # Reset intrusion alert flag if the person leaves all restricted zones
                    elif not is_intruding and data.get('intrusion_alert_triggered', False):
                         id_manager.id_history[our_id]['intrusion_alert_triggered'] = False

            # Check for falling
            for our_id, data in id_manager.id_history.items():
                if len(data['pose_history']) > 10: # Need at least a few frames of pose history
                    is_falling = is_falling_pose(data['pose_history'], h, id_manager) # Pass id_manager

                    if is_falling and not data.get('falling_alert_triggered', False):
                        logging.warning(f"Falling detected for ID {our_id}.")
                        # Trigger alert
                        threading.Thread(target=play_alert_sound).start()
                        log_detection(f"Falling_ID_{our_id}", 1.0, "N/A", frame)
                        id_manager.id_history[our_id]['falling_alert_triggered'] = True
                    # Reset falling alert flag if the person is no longer falling
                    elif data.get('falling_alert_triggered', False) and not is_falling:
                         id_manager.id_history[our_id]['falling_alert_triggered'] = False

            # Check for fighting
            tracked_ids = list(id_manager.id_history.keys())
            for i in range(len(tracked_ids)):
                for j in range(i + 1, len(tracked_ids)):
                    id1 = tracked_ids[i]
                    id2 = tracked_ids[j]
                    data1 = id_manager.id_history[id1]
                    data2 = id_manager.id_history[id2]

                    # Only check for fighting if both individuals have pose history and neither has a fighting alert already triggered
                    if len(data1['pose_history']) > 5 and len(data2['pose_history']) > 5 and \
                       not data1.get('fighting_alert_triggered', False) and not data2.get('fighting_alert_triggered', False):

                        is_fighting = is_fighting_pose(data1['pose_history'], data2['pose_history'], w, h, id_manager) # Pass id_manager

                        if is_fighting:
                            logging.warning(f"Fighting detected between ID {id1} and ID {id2}.")
                            # Trigger alert
                            threading.Thread(target=play_alert_sound).start()
                            log_detection(f"Fighting_ID_{id1}_vs_ID_{id2}", 1.0, "N/A", frame)
                            # Set the fighting alert flag for both individuals
                            id_manager.id_history[id1]['fighting_alert_triggered'] = True
                            id_manager.id_history[id2]['fighting_alert_triggered'] = True

            # Reset fighting alert flag if they are no longer fighting (based on proximity and movement)
            # This is a bit more complex to reset precisely and might need refinement.
            # For now, we rely on the update method's general alert reset.

            # --- Add logic for kicking, punching here ---
            # Check for kicking and punching
            # Gather all current positions
            all_positions = {oid: d['position_history'][-1][1] for oid, d in id_manager.id_history.items() if len(d['position_history']) > 0}

            for our_id, data in id_manager.id_history.items():
                if len(data['pose_history']) > 5:
                    # Exclude self from others_positions
                    others_positions = [pos for oid, pos in all_positions.items() if oid != our_id]
                    is_kicking = is_kicking_pose(data['pose_history'], w, h, id_manager)
                    is_punching = is_punching_pose(data['pose_history'], w, h, id_manager, others_positions=others_positions)

                    if is_kicking and not data.get('kicking_alert_triggered', False):
                        logging.warning(f"Kicking detected for ID {our_id}.")
                        # Trigger alert
                        threading.Thread(target=play_alert_sound).start()
                        log_detection(f"Kicking_ID_{our_id}", 1.0, "N/A", frame)
                        id_manager.id_history[our_id]['kicking_alert_triggered'] = True
                    # Reset kicking alert flag
                    elif data.get('kicking_alert_triggered', False) and not is_kicking:
                         id_manager.id_history[our_id]['kicking_alert_triggered'] = False

                    if is_punching and not data.get('punching_alert_triggered', False):
                        logging.warning(f"Punching detected for ID {our_id}.")
                        # Trigger alert
                        threading.Thread(target=play_alert_sound).start()
                        log_detection(f"Punching_ID_{our_id}", 1.0, "N/A", frame)
                        id_manager.id_history[our_id]['punching_alert_triggered'] = True
                    # Reset punching alert flag
                    elif data.get('punching_alert_triggered', False) and not is_punching:
                         id_manager.id_history[our_id]['punching_alert_triggered'] = False

            # Draw restricted zones on the frame (optional, for visualization)
            for i, zone in enumerate(RESTRICTED_ZONES):
                cv2.polylines(frame, [zone], True, (0, 0, 255), 2) # Draw in red
                # Add zone number label
                M = cv2.moments(zone)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(frame, f'Zone {i+1}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Add FPS counter
            fps = cap.get(cv2.CAP_PROP_FPS)
            # logging.info(f"FPS: {fps:.1f}") # Optional: log FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw pose landmarks for visualization (optional)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Display the frame
            cv2.imshow('CCTV Feed', frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth # Toggle depth visualization
                logging.debug(f"Depth visualization {'enabled' if show_depth else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application shut down.") # Log application shutdown

if __name__ == "__main__":
    main()
