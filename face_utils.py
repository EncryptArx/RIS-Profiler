import os
import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

def setup_face_app():
    print("[DEBUG] Setting up face analysis app...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    print("[DEBUG] Face analysis app setup complete")
    return app

def load_known_faces(app, folder_path="known_faces"):
    print(f"[DEBUG] Starting to load faces from {folder_path}")
    known_encodings = []
    known_names = []

    if not os.path.exists(folder_path):
        print(f"[ERROR] Directory {folder_path} does not exist!")
        return known_encodings, known_names

    files = os.listdir(folder_path)
    print(f"[DEBUG] Found {len(files)} files in directory")
    
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, filename)
            print(f"[DEBUG] Processing file: {filename}")
            img = cv2.imread(path)
            if img is None:
                print(f"[ERROR] Failed to read image: {filename}")
                continue
                
            print(f"[DEBUG] Image shape: {img.shape}")
            faces = app.get(img)
            if faces:
                print(f"[DEBUG] Found {len(faces)} faces in {filename}")
                known_encodings.append(faces[0].embedding)
                known_names.append(os.path.splitext(filename)[0])
                print(f"[DEBUG] Successfully added face encoding for {filename}")
            else:
                print(f"[!] No face found in {filename}")
    
    print(f"[DEBUG] Loaded {len(known_encodings)} known faces")
    return known_encodings, known_names
