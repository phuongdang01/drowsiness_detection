import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import threading
import time
from playsound import playsound
import csv

# --- Khởi tạo Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --- Các hằng số ---
# Mắt (6 điểm)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Miệng
MOUTH = [61, 291, 13, 14]
# Tư thế đầu
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# --- Các hàm tính toán ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[3])  # trên - dưới
    B = distance.euclidean(mouth[0], mouth[1])  # trái - phải
    if B == 0: return 0.0
    return A / B

def play_alert_loop(stop_event):
    while not stop_event.is_set():
        playsound("alert.wav")

# --- Ngưỡng và tham số ---
EAR_THRESH = 0.22
EAR_FRAMES = 15       #tương đương 0.5 giây
MAR_THRESH = 0.6
MAR_FRAMES = 15       #tương đương 0.5 giây
NOD_PITCH_THRESH = 20
NOD_FRAMES = 10

# --- Biến trạng thái ---
eye_counter = 0
yawn_counter = 0
nod_counter = 0
stop_event = threading.Event()
alert_thread = None
prev_time = 0
pitch_history = []

# --- Khởi động camera / video ---
cap = cv2.VideoCapture("kodeokinhcosang.mp4")  # or 0 for webcam
print("🚗 Nhấn Q để thoát...")

# --- Mở file log CSV ---
log = open("threshold_test4.csv", "w", newline="", encoding="utf-8")
writer = csv.writer(log)
writer.writerow(["frame", "EAR", "MAR", "Pitch", "EyesClosed", "Yawning", "HeadNod"])

curr_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    is_alert = False

    if results.multi_face_landmarks:
        # only process first face (you had max_num_faces=1 anyway)
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        def get_xy(i):
            return (int(lm[i].x * w), int(lm[i].y * h))

        # 1️⃣ EAR (Eye Aspect Ratio)
        left_eye = [get_xy(i) for i in LEFT_EYE]
        right_eye = [get_xy(i) for i in RIGHT_EYE]
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 2️⃣ MAR (Mouth Aspect Ratio)
        mouth_pts = [get_xy(61), get_xy(291), get_xy(13), get_xy(14)]
        mar = mouth_aspect_ratio(mouth_pts)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # --- Vẽ khung miệng ---
        mouth_contour = np.array([
            get_xy(61), get_xy(81), get_xy(13), get_xy(311),
            get_xy(291), get_xy(308), get_xy(14), get_xy(78)
        ], np.int32)
        cv2.polylines(frame, [mouth_contour], True, (255, 0, 0), 1)

        # 3️⃣ Head Pose (Pitch)
        image_points = np.array([
            get_xy(1), get_xy(199), get_xy(33),
            get_xy(263), get_xy(61), get_xy(291)
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            P = np.hstack((rotation_matrix, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(P)
            # euler_angles[0] có thể là numpy scalar -> ép về float
            pitch = float(euler_angles[0])
            cv2.putText(frame, f"Pitch: {int(pitch)} deg", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            pitch = 0.0

        # --- Ghi log vào CSV (đảm bảo pitch đã có giá trị) ---
        writer.writerow([
            curr_frame,
            ear,
            mar,
            pitch,
            eye_counter >= EAR_FRAMES,
            yawn_counter >= MAR_FRAMES,
            nod_counter >= NOD_FRAMES
        ])
        curr_frame += 1

        # --- Logic phát hiện ---
        # Nhắm mắt
        if ear < EAR_THRESH:
            eye_counter += 1
        else:
            eye_counter = 0

        # Ngáp
        if mar > MAR_THRESH:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # --- Làm mượt pitch ---
        pitch_history.append(pitch)
        if len(pitch_history) > 5:
            pitch_history.pop(0)
        smooth_pitch = sum(pitch_history) / len(pitch_history)

        # --- Chỉ cho phép head-nod nếu KHÔNG ngáp và KHÔNG nhắm mắt ---
        valid_head_pose = (ear > EAR_THRESH) and (mar < MAR_THRESH)

        # Head nod
        if valid_head_pose and smooth_pitch > NOD_PITCH_THRESH:
            nod_counter += 1
        else:
            nod_counter = 0

        # --- Hiển thị cảnh báo trên frame ---
        if eye_counter >= EAR_FRAMES:
            is_alert = True
            cv2.putText(frame, "⚠️ EYES CLOSED!", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if yawn_counter >= MAR_FRAMES:
            is_alert = True
            cv2.putText(frame, "⚠️ YAWNING!", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if nod_counter >= NOD_FRAMES:
            is_alert = True
            cv2.putText(frame, "⚠️ HEAD NOD!", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- Hiệu ứng nhấp nháy (đỏ ↔ cam) ---
            blink_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 165, 255)

            # --- Vẽ khung quanh khuôn mặt ---
            face_x = int(lm[1].x * w)
            face_y = int(lm[1].y * h)
            box_size = 200
            cv2.rectangle(frame,
                        (face_x - box_size//2, face_y - box_size//2),
                        (face_x + box_size//2, face_y + box_size//2),
                        blink_color, 3)

            # --- Vẽ mũi tên hướng xuống ---
            nose_tip = get_xy(1)
            cv2.arrowedLine(frame,
                            (nose_tip[0], nose_tip[1] - 60),
                            (nose_tip[0], nose_tip[1] + 60),
                            blink_color, 4, tipLength=0.3)

    # --- CẢNH BÁO ÂM THANH ---
    if is_alert:
        cv2.putText(frame, "🚨 DROWSINESS ALERT!", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if alert_thread is None or not alert_thread.is_alive():
            stop_event.clear()
            alert_thread = threading.Thread(target=play_alert_loop, args=(stop_event,), daemon=True)
            alert_thread.start()
    else:
        if alert_thread is not None and alert_thread.is_alive():
            stop_event.set()
            alert_thread = None

    # --- FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Drowsiness Detection (Optimized)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp ---
if alert_thread is not None and alert_thread.is_alive():
    stop_event.set()
cap.release()
cv2.destroyAllWindows()
log.close()
