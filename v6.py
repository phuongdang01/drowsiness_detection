import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
from playsound import playsound
import threading
import math
import time
import os # Thêm thư viện OS để kiểm tra tệp
import sys # THÊM MỚI: Cần thiết cho lệnh sys.exit()

# --- Biến kiểm soát âm thanh ---
# Biến này sẽ giúp chúng ta dừng luồng âm thanh
stop_alert_event = threading.Event()
alert_thread = None
SOUND_FILE = "alert.wav"

# --- Khởi tạo Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# --- Các hằng số cho Landmarks ---
# Mắt (dựa trên 478 điểm của Mediapipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Miệng (cho nhận diện ngáp)
MOUTH_INNER = [13, 14, 61, 291] # Môi trên, Môi dưới, Mép trái, Mép phải
# THÊM MỚI: Viền ngoài của miệng (20 điểm, tương tự dlib)
MOUTH_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 
    84, 181, 91, 146
]

# --- Các hàm tính toán ---

def eye_aspect_ratio(eye_coords):
    """Tính toán EAR dựa trên tọa độ (đã chuẩn hóa hoặc pixel)"""
    # Tính khoảng cách dọc
    A = distance.euclidean(eye_coords[1], eye_coords[5])
    B = distance.euclidean(eye_coords[2], eye_coords[4])
    # Tính khoảng cách ngang
    C = distance.euclidean(eye_coords[0], eye_coords[3])
    
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_coords):
    """Tính toán MAR (Môi trên, Môi dưới, Mép trái, Mép phải)"""
    # Tính khoảng cách dọc
    A = distance.euclidean(mouth_coords[0], mouth_coords[1])
    # Tính khoảng cách ngang
    B = distance.euclidean(mouth_coords[2], mouth_coords[3])
    if B == 0:
        return 0
    mar = A / B
    return mar

# Hàm phát âm thanh lặp lại (trong một luồng riêng)
def play_alert_loop(stop_event_check):
    """Phát tệp âm thanh lặp đi lặp lại cho đến khi sự kiện stop_event được set."""
    while not stop_event_check.is_set():
        try:
            playsound(SOUND_FILE)
            time.sleep(0.5) # Tạm dừng ngắn để tránh chồng chéo âm thanh
        except Exception as e:
            if not stop_event_check.is_set():
                print(f"[LỖI ÂM THANH] Không thể phát '{SOUND_FILE}': {e}")
                print("Vui lòng đảm bảo tệp tồn tại và thư viện 'playsound' hoạt động.")
            break # Thoát khỏi vòng lặp nếu có lỗi

# Hàm quản lý luồng âm thanh
def trigger_alert(start_alert=True):
    """Kích hoạt hoặc dừng luồng cảnh báo."""
    global alert_thread, stop_alert_event

    if start_alert:
        # 1. BẮT ĐẦU CẢNH BÁO
        # Nếu luồng chưa chạy, hãy khởi động nó
        if alert_thread is None or not alert_thread.is_alive():
            print("[CẢNH BÁO] Kích hoạt cảnh báo ngủ gật!")
            stop_alert_event.clear()
            alert_thread = threading.Thread(target=play_alert_loop,
                                            args=(stop_alert_event,),
                                            daemon=True)
            alert_thread.start()
    else:
        # 2. DỪNG CẢNH BÁO
        if alert_thread is not None and alert_thread.is_alive():
            print("[INFO] Tắt cảnh báo.")
            stop_alert_event.set()
            alert_thread = None

# --- Kiểm tra tệp âm thanh trước khi bắt đầu ---
if not os.path.exists(SOUND_FILE):
    print(f"[LỖI] Không tìm thấy tệp âm thanh: '{SOUND_FILE}'")
    print("Chương trình sẽ chạy mà không có âm thanh cảnh báo.")
    
# --- Các ngưỡng (Thresholds) và Biến toàn cục ---

# Ngưỡng EAR (Tỷ lệ khung mắt)
EAR_THRESH = 0.25      # Ngưỡng nhắm mắt (cần tinh chỉnh cho camera của bạn)
EAR_FRAME_CHECK = 15   # Số khung hình liên tiếp để kích hoạt
eye_flag_counter = 0

# Ngưỡng MAR (Ngáp)
MAR_THRESH = 0.5       # Ngưỡng ngáp (cần tinh chỉnh)
YAWN_FRAME_CHECK = 20  # Số khung hình liên tiếp để kích hoạt
yawn_flag_counter = 0

# Ngưỡng Head Nod (Gật đầu)
NOD_PITCH_THRESH = 20  # Độ (cúi xuống 20 độ so với phương ngang)
NOD_FRAME_CHECK = 15   # Số khung hình liên tiếp để kích hoạt
nod_flag_counter = 0

prev_time_fps = 0      # Để tính FPS

# --- Khởi động Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[LỖI] Không thể mở webcam. Vui lòng kiểm tra camera của bạn.")
    sys.exit()

print("📸 Nhấn 'q' để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Kết thúc luồng video.")
        break

    # Lấy kích thước khung hình
    h, w, _ = frame.shape
    if h == 0 or w == 0: continue

    frame = cv2.flip(frame, 1) # Lật ngang (như gương)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý frame với Mediapipe
    results = face_mesh.process(rgb_frame)

    is_drowsy_trigger = False # Biến kiểm soát việc kích hoạt cảnh báo

    if results.multi_face_landmarks:
        # Chỉ lấy khuôn mặt đầu tiên
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark
        
        # Hàm tiện ích để lấy tọa độ pixel (x, y)
        def get_coords(index):
            return (int(lm[index].x * w), int(lm[index].y * h))

        # --- 1. TÍNH TOÁN NHẮM MẮT (EAR) ---
        left_eye_coords = [get_coords(i) for i in LEFT_EYE]
        right_eye_coords = [get_coords(i) for i in RIGHT_EYE]
        
        # Vẽ đa giác quanh mắt
        cv2.polylines(frame, [np.array(left_eye_coords, dtype=np.int32)], True, (0,255,0), 1)
        cv2.polylines(frame, [np.array(right_eye_coords, dtype=np.int32)], True, (0,255,0), 1)

        leftEAR = eye_aspect_ratio(left_eye_coords)
        rightEAR = eye_aspect_ratio(right_eye_coords)
        ear = (leftEAR + rightEAR) / 2.0
        cv2.putText(frame, f"EAR: {ear:.2f}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 2. TÍNH TOÁN NGÁP (MAR) ---
        # Lấy 4 điểm: môi trên (13), môi dưới (14), mép trái (61), mép phải (291)
        mouth_coords = [get_coords(13), get_coords(14), get_coords(61), get_coords(291)]
        mar = mouth_aspect_ratio(mouth_coords)
        cv2.putText(frame, f"MAR: {mar:.2f}", (w - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- THÊM MỚI: Vẽ đường viền miệng (giống code 1) ---
        mouth_outline_coords = [get_coords(i) for i in MOUTH_OUTLINE]
        mouthHull = cv2.convexHull(np.array(mouth_outline_coords, dtype=np.int32))
        # Vẽ đường viền màu vàng (BGR: 0, 255, 255)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        # --- 3. TÍNH TOÁN GẬT ĐẦU (HEAD POSE - PITCH) ---
        # Lấy các điểm 2D trên ảnh
        image_points = np.array([
            get_coords(1),    # Mũi
            get_coords(199),  # Cằm
            get_coords(33),   # Góc mắt trái
            get_coords(263),  # Góc mắt phải
            get_coords(61),   # Mép trái
            get_coords(291)   # Mép phải
        ], dtype="double")
        
        # Mô hình 3D (tọa độ chuẩn - không cần chính xác tuyệt đối)
        model_points = np.array([
            (0.0, 0.0, 0.0),      # Mũi
            (0.0, -330.0, -65.0), # Cằm
            (-225.0, 170.0, -135.0), # Góc mắt trái
            (225.0, 170.0, -135.0),  # Góc mắt phải
            (-150.0, -150.0, -125.0), # Mép trái
            (150.0, -150.0, -125.0)   # Mép phải
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
        (success, rotation_vector, _) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        pitch = 0.0
        if success:
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            # Phân rã ma trận chiếu để lấy góc Euler
            (_, _, _, _, _, _, euler_angles) = cv2.decomposeProjectionMatrix(
                np.hstack((rotation_matrix, np.zeros((3, 1))))
            )
            # SỬA LỖI: euler_angles[0] là một mảng (ví dụ: [20.5]).
            # Chúng ta cần lấy giá trị float bên trong nó bằng [0][0]
            # trước khi format bằng f-string ".:2f".
            pitch = euler_angles[0][0]
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (w - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- LOGIC CỜ (FLAG) ---
        
        # Cờ nhắm mắt
        if ear < EAR_THRESH:
            eye_flag_counter += 1
        else:
            eye_flag_counter = 0
        
        # Cờ ngáp
        if mar > MAR_THRESH:
            yawn_flag_counter += 1
        else:
            yawn_flag_counter = 0
        
        # Cờ gật đầu
        if pitch > NOD_PITCH_THRESH:
            nod_flag_counter += 1
        else:
            nod_flag_counter = 0

        # --- KIỂM TRA KÍCH HOẠT CẢNH BÁO ---
        if eye_flag_counter >= EAR_FRAME_CHECK:
            is_drowsy_trigger = True
            cv2.putText(frame, "MAT NHAM", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if yawn_flag_counter >= YAWN_FRAME_CHECK:
            is_drowsy_trigger = True
            cv2.putText(frame, "DANG NGAP", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawn_flag_counter = 0 # Reset để phát hiện ngáp lần nữa
        
        if nod_flag_counter >= NOD_FRAME_CHECK:
            is_drowsy_trigger = True
            cv2.putText(frame, "GAT DAU", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            nod_flag_counter = 0 # Reset

    else:
        # Nếu không phát hiện khuôn mặt, reset tất cả các cờ
        eye_flag_counter = 0
        yawn_flag_counter = 0
        nod_flag_counter = 0

    # --- QUẢN LÝ CẢNH BÁO (NGOÀI VÒNG LẶP LANDMARKS) ---
    
    if is_drowsy_trigger:
        cv2.putText(frame, "!!! CANH BAO NGU GAT !!!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Bắt đầu phát âm thanh (nếu tệp tồn tại)
        if os.path.exists(SOUND_FILE):
            trigger_alert(start_alert=True)
    else:
        # Dừng âm thanh (chỉ khi mắt đã mở lại VÀ không có cờ nào khác)
        # Chúng ta dùng `ear` từ vòng lặp trước, hoặc reset nếu không có mặt
        ear_check = locals().get('ear', 0.0) # Lấy giá trị 'ear' nếu tồn tại
        if ear_check >= EAR_THRESH and eye_flag_counter == 0:
             trigger_alert(start_alert=False)

    # TÍNH FPS
    curr_time_fps = time.time()
    if curr_time_fps != prev_time_fps:
        fps = 1 / (curr_time_fps - prev_time_fps)
        prev_time_fps = curr_time_fps
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("He thong Phat hien Ngu gat (Mediapipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- DỌN DẸP TRƯỚC KHI THOÁT ---
print("[INFO] Đang dọn dẹp và thoát...")
trigger_alert(start_alert=False) # Đảm bảo luồng âm thanh đã tắt
cap.release()
cv2.destroyAllWindows()
face_mesh.close()