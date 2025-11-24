📌 FallGuard AI – Real-Time Fall Detection Using MediaPipe & OpenCV

A lightweight, real-time fall detection system using MediaPipe Pose and OpenCV.
The system analyzes the human torso angle and shoulder–hip alignment to detect:

🟢 Stable posture

🟡 Suspicious posture

🟠 Lying (possible fall)

🔴 FALL DETECTED (lying for more than threshold seconds)

This project works with a webcam, image, or video file.

📁 Project Structure
├── README.md                # Documentation
├── main.py                  # Main fall detection script
├── pose_detection.png       # Sample image / demo screenshot
└── video.mp4                # Sample video for testing

🚀 Features

Real-time pose detection

Torso angle & shoulder–hip vertical distance calculation

Color-coded fall warnings

Timer-based fall confirmation

Works with webcam / video file

Lightweight (MediaPipe + OpenCV only)

📦 Installation
1. Clone the repository
git clone https://github.com/Omvrat96/FallGuardAI.git
cd FallGuardAI

2. Install dependencies
pip install opencv-python mediapipe

▶️ Usage
Run with webcam
python main.py --source 0

Run with a video file
python main.py --source video.mp4

Change fall confirmation duration (default: 2 sec)
python main.py --source 0 --fall_time 3.5

🧠 How It Works

MediaPipe Pose extracts human keypoints.

Midpoints of shoulders and hips are computed.

Torso angle is calculated using the shoulder→hip vector.

Classification rules:

Angle > 55° or shoulders below hips → lying (possible fall)  
Angle > 25° → suspicious  
Else → stable  


If lying posture continues past fall_time, system triggers FALL DETECTED.

🖼️ Example Output

🔧 Customization

You can modify:

Angle thresholds

Fall duration

Overlay colors

Add warning sounds/alerts

Add YOLO person tracking

Tell me if you'd like any upgrades!

🧩 Requirements

Python 3.7+

OpenCV

MediaPipe
