import cv2
import mediapipe as mp
import math

# --- Hàm tính khoảng cách giữa 2 điểm ---
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# --- Hàm tính Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(landmarks, eye_indices):
    # 6 điểm chính của mắt (trên - dưới - trái - phải)
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    # khoảng cách dọc
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    # khoảng cách ngang
    horizontal = euclidean_distance(p1, p4)
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# --- Các điểm landmark quanh mắt trái / phải ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- Khởi tạo MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Mở webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Tính EAR cho 2 mắt
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # Vẽ vòng quanh mắt
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Kiểm tra nhắm mắt ---
            if ear < 0.20:
                cv2.putText(frame, "Eyes Closed 👀", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Eyes Open", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Hiển thị giá trị EAR (debug)
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
