import os
import csv
from datetime import datetime
import cv2
import logging
import numpy as np

def log_detection(name, similarity, liveness, frame=None):
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/captured_frames', exist_ok=True)
    log_file = 'data/logs.csv'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, name, f'{similarity:.2f}', liveness]
    # Save frame if provided and valid
    if frame is not None:
        logging.debug("Frame is not None in log_detection.")
        # Add check for valid frame
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            logging.error("Invalid or empty frame passed to log_detection.")
            row.append('Invalid frame')
        else:
            frame_filename = f'data/captured_frames/{name}_{timestamp.replace(":", "-").replace(" ", "_")}.jpg'
            logging.debug(f"Attempting to save frame to: {frame_filename}")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                logging.debug(f"Successfully saved frame to {frame_filename}")
                row.append(frame_filename)
            else:
                logging.error(f"Failed to save frame to {frame_filename}")
                row.append('Save failed')
    else:
        row.append('')
    # Write to CSV
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'name', 'similarity', 'liveness', 'frame_path'])
        writer.writerow(row) 