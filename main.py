"""
Simple Fall Detection using MediaPipe + OpenCV
- Very small, easy-to-read script
- Input: webcam (default) or video/image file (--source)
- Output: shows window with label: Stable / Suspicious / FALL DETECTED
- Install: pip install mediapipe opencv-python
"""

import cv2
import mediapipe as mp
import time
import argparse
import math

mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe landmark indices
L_SH = 11
R_SH = 12
L_HIP = 23
R_HIP = 24

def torso_info(landmarks, w, h):
    """Return shoulder_mid_y, hip_mid_y, torso_angle_deg (0 vertical, ~90 horizontal)."""
    lsh = landmarks[L_SH]
    rsh = landmarks[R_SH]
    lhp = landmarks[L_HIP]
    rhp = landmarks[R_HIP]

    sx = (lsh.x + rsh.x) / 2.0 * w
    sy = (lsh.y + rsh.y) / 2.0 * h
    hx = (lhp.x + rhp.x) / 2.0 * w
    hy = (lhp.y + rhp.y) / 2.0 * h

    dx = hx - sx
    dy = hy - sy
    # angle between shoulder->hip vector and vertical axis
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-7))
    return sy, hy, angle

def classify_and_label(sy, hy, angle, img_h):
    """
    Simple heuristics:
      - If shoulder below/equal hip or angle > 55 -> likely lying
      - If angle > 25 or shoulder-hip distance small -> suspicious
      - Else stable
    """
    vdist_norm = abs(sy - hy) / img_h
    if sy >= hy or angle > 55 or vdist_norm < 0.03:
        return "lying"
    if angle > 25 or vdist_norm < 0.08:
        return "suspicious"
    return "stable"

def main(source, fall_time):
    # open source (camera or file)
    try:
        cap = cv2.VideoCapture(int(source))
    except:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Cannot open source:", source)
        return

    state = "stable"
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = POSE.process(rgb)

        label = "No person"
        color = (200, 200, 200)  # gray
        timer_text = ""

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # compute torso info; catch if landmarks missing
            try:
                sy, hy, angle = torso_info(lm, w, h)
            except Exception:
                sy = hy = angle = None

            if sy is not None:
                cat = classify_and_label(sy, hy, angle, h)

                if cat == "stable":
                    state = "stable"
                    start_time = None
                    label = "Stable"
                    color = (0, 200, 0)
                elif cat == "suspicious":
                    # short transient label; reset fall timer
                    state = "suspicious"
                    start_time = None
                    label = "Suspicious Posture"
                    color = (0, 200, 200)
                else:  # lying candidate
                    # start or continue timer
                    if start_time is None:
                        start_time = time.time()
                    elapsed = time.time() - start_time
                    if elapsed >= fall_time:
                        label = f"FALL DETECTED ({int(elapsed)}s)"
                        color = (0, 0, 255)
                    else:
                        label = f"Suspicious/Lying ({int(elapsed)}s)"
                        color = (0, 140, 255)

                    state = "lying"

                timer_text = f"angle={angle:.1f}"

                # draw simple torso line + points
                sx = int((lm[L_SH].x + lm[R_SH].x) / 2 * w)
                sy_i = int((lm[L_SH].y + lm[R_SH].y) / 2 * h)
                hx = int((lm[L_HIP].x + lm[R_HIP].x) / 2 * w)
                hy_i = int((lm[L_HIP].y + lm[R_HIP].y) / 2 * h)
                cv2.line(frame, (sx, sy_i), (hx, hy_i), color, 3)
                cv2.circle(frame, (sx, sy_i), 5, color, -1)
                cv2.circle(frame, (hx, hy_i), 5, color, -1)

            else:
                label = "Person (no torso)"
        # overlay text
        cv2.rectangle(frame, (10, 10), (360, 80), (0,0,0), -1)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if timer_text:
            cv2.putText(frame, timer_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow("Simple Fall Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="0 for webcam or path to video/image")
    p.add_argument("--fall_time", type=float, default=2.0,
                   help="seconds of lying before declaring fall")
    args = p.parse_args()
    main(args.source, args.fall_time)