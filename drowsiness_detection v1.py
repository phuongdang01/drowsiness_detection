import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
from playsound import playsound
import threading

# Khởi tạo Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Danh sách các điểm quanh mắt (Mediapipe index)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Hàm này sẽ lặp lại âm thanh cho đến khi nhận được tín hiệu dừng
def play_alert_loop(stop_event):
    while not stop_event.is_set():
        playsound("alert.wav")
        # Bạn có thể thêm một khoảng nghỉ ngắn ở đây nếu muốn
        # time.sleep(0.1) 

thresh = 0.25
frame_check = 20
flag = 0
# Biến để kiểm soát luồng âm thanh
stop_alert_event = threading.Event()
alert_thread = None

cap = cv2.VideoCapture(0)
print("📸 Nhấn Q để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            lm = face_landmarks.landmark

            left_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]

            cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0,255,0), 1)
            cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0,255,0), 1)

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Nếu luồng âm thanh chưa chạy, hãy khởi động nó
                    if alert_thread is None or not alert_thread.is_alive():
                        stop_alert_event.clear() # Đảm bảo tín hiệu dừng đã tắt
                        alert_thread = threading.Thread(target=play_alert_loop, 
                                                        args=(stop_alert_event,), 
                                                        daemon=True)
                        alert_thread.start()
            else:
                # Mắt đã mở!
                flag = 0
                # Nếu luồng âm thanh đang chạy, hãy gửi tín hiệu dừng
                if alert_thread is not None and alert_thread.is_alive():
                    stop_alert_event.set() # Gửi tín hiệu dừng
                    alert_thread = None # Reset biến luồng
            
    cv2.imshow("Drowsiness Detection (With Sound)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp trước khi thoát
if alert_thread is not None and alert_thread.is_alive():
    stop_alert_event.set() # Dừng luồng nếu chương trình thoát

cap.release()
cv2.destroyAllWindows()