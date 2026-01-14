import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
from playsound import playsound
import threading
import math # Cần cho tính toán góc

# --- Khởi tạo Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Các hằng số cho Landmarks ---
# Mắt
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Miệng (cho nhận diện ngáp)
MOUTH_OUTER = [61, 291, 0, 17] # Trái, Phải, Trên, Dưới (chỉ để tham khảo)
MOUTH_INNER = [78, 308, 13, 14] # Tương tự, cho MAR
# Điểm landmarks cho tư thế đầu (Head Pose)
HEAD_POSE_LANDMARKS = [
    33, 263, 1, 61, 291, 199 # Mắt trái, Mắt phải, Mũi, Miệng trái, Miệng phải, Cằm
]

# --- Các hàm tính toán ---

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Hàm tính Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    # Tính khoảng cách dọc (môi trên và môi dưới)
    A = distance.euclidean(mouth[0], mouth[1]) # Ví dụ: (13, 14)
    # Tính khoảng cách ngang (2 mép)
    B = distance.euclidean(mouth[2], mouth[3]) # Ví dụ: (61, 291)
    if B == 0: # Tránh chia cho 0
        return 0
    mar = A / B
    return mar

# Hàm phát âm thanh lặp lại
def play_alert_loop(stop_event):
    while not stop_event.is_set():
        playsound("alert.wav")

# --- Các ngưỡng (Thresholds) và Biến toàn cục ---

# Ngưỡng EAR
EAR_THRESH = 0.25
EAR_FRAME_CHECK = 20
eye_flag = 0

# Ngưỡng MAR (Ngáp)
MAR_THRESH = 0.5 # Ngưỡng này cần được tinh chỉnh
YAWN_FRAME_CHECK = 10
yawn_flag = 0

# Ngưỡng Head Nod (Gật đầu)
NOD_PITCH_THRESH = 20 # Độ (cúi xuống 20 độ)
NOD_FRAME_CHECK = 15
nod_flag = 0

# Biến kiểm soát luồng âm thanh
stop_alert_event = threading.Event()
alert_thread = None

# --- Khởi động Camera ---
cap = cv2.VideoCapture(0)
print("📸 Nhấn Q để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy kích thước khung hình
    h, w, _ = frame.shape

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý frame với Mediapipe
    results = face_mesh.process(rgb)

    ear = 0.0 # Khởi tạo ear
    is_trigger_alert = False # Biến kiểm soát việc kích hoạt cảnh báo

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            
            # Chuyển đổi landmarks sang tọa độ pixel (x, y)
            def get_coords(index):
                return (int(lm[index].x * w), int(lm[index].y * h))
            
            # 1. TÍNH TOÁN NHẮM MẮT (EAR)
            left_eye_coords = [get_coords(i) for i in LEFT_EYE]
            right_eye_coords = [get_coords(i) for i in RIGHT_EYE]
            
            cv2.polylines(frame, [np.array(left_eye_coords, dtype=np.int32)], True, (0,255,0), 1)
            cv2.polylines(frame, [np.array(right_eye_coords, dtype=np.int32)], True, (0,255,0), 1)

            leftEAR = eye_aspect_ratio(left_eye_coords)
            rightEAR = eye_aspect_ratio(right_eye_coords)
            ear = (leftEAR + rightEAR) / 2.0

            # 2. TÍNH TOÁN NGÁP (MAR)
            # Lấy 4 điểm: môi trên (13), môi dưới (14), mép trái (61), mép phải (291)
            mouth_coords = [get_coords(13), get_coords(14), get_coords(61), get_coords(291)]
            mar = mouth_aspect_ratio(mouth_coords)

            # 3. TÍNH TOÁN GẬT ĐẦU (HEAD POSE - PITCH)
            # Lấy các điểm 2D trên ảnh
            image_points = np.array([
                get_coords(1),    # Mũi
                get_coords(199),  # Cằm
                get_coords(33),   # Góc mắt trái
                get_coords(263),  # Góc mắt phải
                get_coords(61),   # Mép trái
                get_coords(291)   # Mép phải
            ], dtype="double")
            
            # Mô hình 3D (tọa độ chuẩn)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Mũi
                (0.0, -330.0, -65.0),        # Cằm
                (-225.0, 170.0, -135.0),     # Góc mắt trái
                (225.0, 170.0, -135.0),      # Góc mắt phải
                (-150.0, -150.0, -125.0),    # Mép trái
                (150.0, -150.0, -125.0)      # Mép phải
            ])
            
            # Thông số camera (giả định)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1)) # Không có méo ống kính

            # Giải PnP để tìm tư thế
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            #
            # <<< PHẦN SỬA LỖI SỐ 2 BẮT ĐẦU TẠI ĐÂY >>>
            #
            # Chỉ tính toán góc nếu solvePnP thành công
            if success:
                # Lấy góc Pitch (cúi/ngửa)
                (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
                P_mat = np.hstack((rotation_matrix, translation_vector))
                (_, _, _, _, _, _, euler_angles) = cv2.decomposeProjectionMatrix(P_mat)
                pitch = euler_angles[0]
                
                # Cập nhật cờ gật đầu
                if pitch > NOD_PITCH_THRESH:
                    nod_flag += 1
                else:
                    nod_flag = 0
            else:
                # Nếu PnP thất bại, reset cờ
                nod_flag = 0
            #
            # <<< KẾT THÚC PHẦN SỬA LỖI SỐ 2 >>>
            #

            # --- LOGIC KIỂM TRA TỔNG HỢP ---
            
            # Cập nhật cờ (flag) cho mắt và ngáp
            if ear < EAR_THRESH:
                eye_flag += 1
            else:
                eye_flag = 0
            
            if mar > MAR_THRESH:
                yawn_flag += 1
            else:
                yawn_flag = 0

            # (Logic cờ gật đầu đã được chuyển vào khối 'if success' ở trên)

            # Kiểm tra xem có kích hoạt cảnh báo không
            if eye_flag >= EAR_FRAME_CHECK:
                is_trigger_alert = True
                cv2.putText(frame, "EYES CLOSED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if yawn_flag >= YAWN_FRAME_CHECK:
                is_trigger_alert = True
                cv2.putText(frame, "YAWNING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn_flag = 0 # Reset để có thể phát hiện ngáp lần nữa
            
            if nod_flag >= NOD_FRAME_CHECK:
                is_trigger_alert = True
                cv2.putText(frame, "HEAD NOD", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                nod_flag = 0 # Reset để phát hiện gật đầu lần nữa

    else:
        # <<< THÊM MỚI (QUAN TRỌNG) >>>
        # Nếu không phát hiện khuôn mặt, hãy reset tất cả các cờ
        # Điều này ngăn việc báo động sai khi khuôn mặt xuất hiện trở lại
        eye_flag = 0
        yawn_flag = 0
        nod_flag = 0


    # --- LOGIC CẢNH BÁO (BÊN NGOÀI VÒNG LẶP LANDMARKS) ---
    
    # 1. BẮT ĐẦU CẢNH BÁO
    # Nếu bất kỳ cờ nào được kích hoạt (mắt nhắm, ngáp, hoặc gật đầu)
    if is_trigger_alert:
        cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Nếu luồng âm thanh chưa chạy, hãy khởi động nó
        if alert_thread is None or not alert_thread.is_alive():
            stop_alert_event.clear()
            alert_thread = threading.Thread(target=play_alert_loop, 
                                            args=(stop_alert_event,), 
                                            daemon=True)
            alert_thread.start()

    # 2. DỪNG CẢNH BÁO
    # Chỉ dừng lại khi mắt mở (ear > thresh VÀ eye_flag đã reset về 0)
    # Logic này vẫn đúng: nếu không có mặt, ear = 0.0, báo động sẽ không dừng
    if ear >= EAR_THRESH and eye_flag == 0:
        if alert_thread is not None and alert_thread.is_alive():
            stop_alert_event.set()
            alert_thread = None

    # Hiển thị frame
    cv2.imshow("Drowsiness Detection v2 (Eyes, Yawn, Nod)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp trước khi thoát
if alert_thread is not None and alert_thread.is_alive():
    stop_alert_event.set()

cap.release()
cv2.destroyAllWindows()