import pandas as pd
import os

# Path to logs and known faces
data_dir = 'data'
logs_path = os.path.join(data_dir, 'logs.csv')
known_faces_dir = 'known_faces'

# Load logs
logs = pd.read_csv(logs_path)

# Get list of known face names (without extension)
known_face_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
known_names = set(os.path.splitext(f)[0] for f in known_face_files)

# Face Recognition Accuracy
face_logs = logs[logs['name'].notna() & (logs['name'] != '') & (logs['name'] != 'Unknown') & (~logs['name'].str.contains('Loitering_ID'))]
total_faces = len(face_logs)
correct_faces = sum(face_logs['name'].isin(known_names))
face_accuracy = (correct_faces / total_faces) * 100 if total_faces > 0 else 0

# Liveness Detection Accuracy (with ground truth if available)
liveness_logs = logs[logs['liveness'].notna() & (logs['liveness'] != 'N/A')]
total_liveness = len(liveness_logs)

if 'ground_truth_liveness' in logs.columns:
    # Compare system prediction to ground truth
    valid_liveness = liveness_logs[liveness_logs['ground_truth_liveness'].notna()]
    correct_liveness = sum(valid_liveness['liveness'].str.lower() == valid_liveness['ground_truth_liveness'].str.lower())
    total_liveness = len(valid_liveness)
    liveness_accuracy = (correct_liveness / total_liveness) * 100 if total_liveness > 0 else 0
    liveness_note = '(using ground truth)'
else:
    # Fallback: assume all should be "Live"
    correct_liveness = sum(liveness_logs['liveness'] == 'Live')
    liveness_accuracy = (correct_liveness / total_liveness) * 100 if total_liveness > 0 else 0
    liveness_note = '(no ground truth column found)'

# Behavior Analysis (Loitering Detection)
loitering_logs = logs[logs['name'].str.contains('Loitering_ID', na=False)]
total_loitering = len(loitering_logs)

# Print results
print('--- Evaluation Results ---')
print(f'Total Recognized Faces: {total_faces}')
print(f'Correctly Recognized Faces: {correct_faces}')
print(f'Face Recognition Accuracy: {face_accuracy:.2f}%')
print()
print(f'Total Liveness Checks: {total_liveness}')
print(f'Correct Liveness: {correct_liveness}')
print(f'Liveness Detection Accuracy: {liveness_accuracy:.2f}% {liveness_note}')
print()
print(f'Total Loitering Events Detected: {total_loitering}')
print('--------------------------')

# Optionally: Save results to a file
with open('evaluation_summary.txt', 'w') as f:
    f.write('--- Evaluation Results ---\n')
    f.write(f'Total Recognized Faces: {total_faces}\n')
    f.write(f'Correctly Recognized Faces: {correct_faces}\n')
    f.write(f'Face Recognition Accuracy: {face_accuracy:.2f}%\n\n')
    f.write(f'Total Liveness Checks: {total_liveness}\n')
    f.write(f'Correct Liveness: {correct_liveness}\n')
    f.write(f'Liveness Detection Accuracy: {liveness_accuracy:.2f}% {liveness_note}\n\n')
    f.write(f'Total Loitering Events Detected: {total_loitering}\n')
    f.write('--------------------------\n') 