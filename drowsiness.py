import cv2
import mediapipe as mp
import numpy as np
import os
import time

# -------------------- AUTO CAMERA DETECTION --------------------
cap = None
for i in [0, 1, 2]:
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"Camera opened at index {i}")
        break

if cap is None:
    print("ERROR: No camera found")
    exit()

# -------------------- MediaPipe Setup --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -------------------- Thresholds --------------------
EAR_THRESHOLD = 0.25
EYE_CLOSED_THRESHOLD = 15
EYE_CLOSED_FRAMES = 0

# Voice alert control
ALERT_COOLDOWN = 5  # seconds
last_alert_time = 0

# -------------------- Eye landmark indices --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------- EAR Calculation --------------------
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received from camera")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # ---------------- FACE BOUNDING BOX ----------------
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, "FACE", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # ---------------- EYE LANDMARKS ----------------
        left_eye = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE],
            dtype=np.int32
        )
        right_eye = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE],
            dtype=np.int32
        )

        # ---------------- EAR ----------------
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < EAR_THRESHOLD:
            color = (0, 0, 255)  # RED
            EYE_CLOSED_FRAMES += 1
        else:
            color = (0, 255, 0)  # GREEN
            EYE_CLOSED_FRAMES = 0

        # ---------------- DRAW EYES ----------------
        cv2.polylines(frame, [left_eye], True, color, 2)
        cv2.polylines(frame, [right_eye], True, color, 2)

        lx, ly, lw, lh = cv2.boundingRect(left_eye)
        rx, ry, rw, rh = cv2.boundingRect(right_eye)
        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, 1)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 1)

        # ---------------- ALERT ----------------
        if EYE_CLOSED_FRAMES >= EYE_CLOSED_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                os.system('say "Alert. Driver is drowsy."')
                last_alert_time = current_time

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
