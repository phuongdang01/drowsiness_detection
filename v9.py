from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
import threading
import simpleaudio
import time

# ============================================================
# 1. HÀM TÍNH EAR / MAR
# ============================================================

def compute_ear(pts):
    p2, p6, p3, p5, p1, p4 = pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C)

def compute_mar(top_lip_pts, bottom_lip_pts):
    T2, T3, T4, T5 = top_lip_pts
    B2, B3, B4, B5 = bottom_lip_pts
    A = np.linalg.norm(T2 - B2)
    B = np.linalg.norm(T3 - B3)
    C = np.linalg.norm(T4 - B4)
    D = np.linalg.norm(T5 - B5)
    return (A + B + C + D) / 4.0

# ============================================================
# 2. LẤY THRESHOLD / SỐ FRAME NGƯỠNG
# ============================================================

print("=== NHẬP NGƯỠNG ===")
EAR_THRESH = float(input("Nhập EAR threshold (vd 0.25): "))
EAR_FRAMES = int(input("Số frame để kết luận ngủ gật (vd 15): "))

MAR_THRESH = float(input("Nhập MAR threshold (vd 0.7): "))
MAR_FRAMES = int(input("Số frame ngáp liên tục (vd 7): "))

NOD_PITCH_THRESH = float(input("Pitch threshold gật đầu (vd 12): "))
NOD_FRAMES = int(input("Số frame gật đầu liên tục (vd 5): "))

output_csv = input("Nhập tên file CSV output: ")

# ============================================================
# 3. CẤU HÌNH MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark mắt
LEFT_EYE = [33, 160, 158, 153, 144, 133]
RIGHT_EYE = [263, 387, 385, 380, 373, 362]

TOP_LIP = [13, 310, 311, 312]
BOTTOM_LIP = [14, 87, 178, 88]

# ============================================================
# 4. SOLVEPnP 6 điểm để tính pitch
# ============================================================

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),       # Mũi
    (-30.0, -125.0, -30.0),
    (30.0, -125.0, -30.0),
    (-60.0, -70.0, -60.0),
    (60.0, -70.0, -60.0),
    (0.0, -150.0, -25.0)
], dtype=np.float32)

FACE_LANDMARK_IDX = [1, 33, 263, 61, 291, 199]
pitch_history = deque(maxlen=10)

def compute_pitch(face_landmarks, w, h):
    IMAGE_POINTS = np.array([
        [face_landmarks[1].x * w, face_landmarks[1].y * h],
        [face_landmarks[33].x * w, face_landmarks[33].y * h],
        [face_landmarks[263].x * w, face_landmarks[263].y * h],
        [face_landmarks[61].x * w, face_landmarks[61].y * h],
        [face_landmarks[291].x * w, face_landmarks[291].y * h],
        [face_landmarks[199].x * w, face_landmarks[199].y * h]
    ], dtype=np.float32)

    focal = w
    cam_matrix = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, IMAGE_POINTS, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0, 0, 0

    rot_mtx, _ = cv2.Rodrigues(rvec)
    pose_mtx = np.hstack((rot_mtx, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mtx)

    pitch = float(euler[0])
    yaw = float(euler[1])
    roll = float(euler[2])

    # fix wrap-around
    if pitch > 180: pitch -= 360
    if pitch < -180: pitch += 360
    if yaw > 180: yaw -= 360
    if yaw < -180: yaw += 360
    if roll > 180: roll -= 360
    if roll < -180: roll += 360

    # EMA smoothing
    if len(pitch_history) == 0:
        smooth_pitch = pitch
    else:
        smooth_pitch = 0.3 * pitch + 0.7 * pitch_history[-1]

    pitch_history.append(smooth_pitch)
    return smooth_pitch, yaw, roll

# ============================================================
# 5. ALARM SYSTEM KHÔNG BỊ BLOCK
# ============================================================

alarm_playing = False
alarm_stop = threading.Event()

def alarm_thread():
    global alarm_playing
    wave = simpleaudio.WaveObject.from_wave_file("alert.wav")
    while not alarm_stop.is_set():
        play = wave.play()
        play.wait_done()

# ============================================================
# 6. MỞ VIDEO + CSV
# ============================================================

cap = cv2.VideoCapture(0)

with open(output_csv, "w") as f:
    f.write("Frame,EAR,MAR,PITCH,EyeClose,Yawn,Nod\n")

curr_frame = 0

eye_cnt = 0
yawn_cnt = 0
nod_cnt = 0

# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    EAR = 0
    MAR = 0
    pitch = 0
    yaw = 0
    roll = 0

    eye_event = False
    yawn_event = False
    nod_event = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        def xy(id):
            return np.array([lm[id].x * w, lm[id].y * h])

        left_pts = np.array([xy(i) for i in LEFT_EYE])
        right_pts = np.array([xy(i) for i in RIGHT_EYE])

        EAR = (compute_ear(left_pts) + compute_ear(right_pts)) / 2

        top_pts = np.array([xy(i) for i in TOP_LIP])
        bot_pts = np.array([xy(i) for i in BOTTOM_LIP])
        MAR = compute_mar(top_pts, bot_pts)

        pitch, yaw, roll = compute_pitch(lm, w, h)

        # =====================================================
        # UPDATE COUNTERS AFTER COMPUTATION
        # =====================================================
        if EAR < EAR_THRESH:
            eye_cnt += 1
        else:
            eye_cnt = 0

        if MAR > MAR_THRESH:
            yawn_cnt += 1
        else:
            yawn_cnt = 0

        valid_head_pose = (EAR > EAR_THRESH) and (MAR < MAR_THRESH) \
                          and abs(yaw) < 25 and abs(roll) < 25

        if valid_head_pose and pitch > NOD_PITCH_THRESH:
            nod_cnt += 1
        else:
            nod_cnt = 0

        # =====================================================
        # DETECT EVENTS
        # =====================================================
        eye_event = (eye_cnt >= EAR_FRAMES)
        yawn_event = (yawn_cnt >= MAR_FRAMES)
        nod_event = (nod_cnt >= NOD_FRAMES)

        # =====================================================
        # ALARM CONTROL
        # =====================================================
        if eye_event or yawn_event or nod_event:
            if not alarm_playing:
                alarm_playing = True
                alarm_stop.clear()
                threading.Thread(target=alarm_thread, daemon=True).start()
        else:
            if alarm_playing:
                alarm_playing = False
                alarm_stop.set()

    # ============================================================
    # WRITE CSV (sau khi cập nhật event)
    # ============================================================
    with open(output_csv, "a") as f:
        f.write(f"{curr_frame},{EAR},{MAR},{pitch},{eye_event},{yawn_event},{nod_event}\n")

    curr_frame += 1

    # SHOW
    cv2.putText(frame, f"EAR={EAR:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"MAR={MAR:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Pitch={pitch:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEAN UP
alarm_stop.set()
cap.release()
cv2.destroyAllWindows()
