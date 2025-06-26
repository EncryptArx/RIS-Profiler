from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Configuration
VIDEO_SOURCE = 0  # Default webcam
LOGS_PATH = "../data/logs.csv"
CAPTURED_FRAMES_PATH = "../data/captured_frames"

def get_camera():
    return cv2.VideoCapture(VIDEO_SOURCE)

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    try:
        if not os.path.exists(LOGS_PATH):
            return jsonify([])  # No alerts if file does not exist
        df = pd.read_csv(LOGS_PATH)
        df = df.dropna()  # Drop rows with any NaN values
        recent_alerts = df.tail(10).to_dict('records')
        return jsonify(recent_alerts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/captured_frames')
def get_captured_frames():
    try:
        frames = []
        for filename in os.listdir(CAPTURED_FRAMES_PATH):
            if filename.endswith(('.jpg', '.png')):
                frames.append({
                    'filename': filename,
                    'url': f'/static/captured_frames/{filename}',
                    'timestamp': datetime.fromtimestamp(
                        os.path.getctime(os.path.join(CAPTURED_FRAMES_PATH, filename))
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })
        return jsonify(frames)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/captured_frames/<path:filename>')
def custom_captured_frames(filename):
    return send_from_directory('../data/captured_frames', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 