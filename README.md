# Face Alert System

Empowering CCTV with AI for Intelligent Surveillance

## Overview
This project brings real-time AI/ML analytics to CCTV systems, enabling facial recognition, liveness (anti-spoofing), behavior analysis (loitering, intrusion, etc.), and instant alerts. It features a web dashboard for monitoring and is optimized for edge deployment.

## Features
- Real-time face recognition using YOLOv8 and InsightFace
- Liveness/anti-spoofing detection
- Behavior analysis: loitering, intrusion, running, falling, fighting, etc.
- Audio and web-based alerts
- Logging and event capture
- Modular, extensible design

## Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/EncryptArx/Face-Alert-System.git
   cd RIS-Profiler
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download or place your YOLOv8 model weights (e.g., yolov8n.pt) in the project root.**
5. **Add known faces to the `known_faces/` directory.**
6. **(Optional) Place test videos in a `sample/` directory.**

## Usage
- **Run the main application:**
  ```sh
  python main.py
  ```
- **To process a video file:**
  Edit `main.py` to set the video path in `cv2.VideoCapture()`.
- **View logs and captured frames in the `data/` directory.**
- **Run evaluation:**
  ```sh
  python evaluate.py
  ```

## Model Weights

- The YOLOv8 model weights (`yolov8n.pt`) will be automatically downloaded by the code if not present.
- If you want to use a custom or pre-downloaded model, download it from [this Google Drive link](https://drive.google.com/file/d/1xOVek65y7D3s3gCs68otCX_nrclg4fYy/view?usp=sharing) and place it in the project root directory.

## Contribution
- Fork the repo and submit pull requests.
- Please update the README and .gitignore as needed.

## License
This project is licensed under the MIT License.

## Repository

- [GitHub: EncryptArx/RIS-Profiler](https://github.com/EncryptArx/RIS-Profiler.git) 
