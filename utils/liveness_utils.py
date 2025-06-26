import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = None

def setup_liveness():
    global face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def is_blinking(frame):
    global face_mesh
    if face_mesh is None:
        setup_liveness()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return False
    # Simple blink detection: check EAR (eye aspect ratio) for left and right eyes
    # Use landmarks: left eye [33, 160, 158, 133, 153, 144], right eye [362, 385, 387, 263, 373, 380]
    def eye_aspect_ratio(eye):
        import numpy as np
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)
    h, w, _ = frame.shape
    for face_landmarks in results.multi_face_landmarks:
        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
        left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        EAR_THRESH = 0.21
        if left_ear < EAR_THRESH or right_ear < EAR_THRESH:
            return True
    return False 