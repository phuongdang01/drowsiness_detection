import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import csv

# --- HÀM TÍNH EAR (giữ nguyên) ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- HÀM TÍNH MAR (MỚI) ---
def mouth_aspect_ratio(mouth):
    # Lấy các điểm mốc cho miệng
    # Khoảng cách dọc
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    # Khoảng cách ngang
    D = distance.euclidean(mouth[12], mouth[16])
    
    mar = (A + B + C) / (3.0 * D)
    return mar

# --- HÀM TÍNH TƯ THẾ ĐẦU (MỚI) ---
def get_head_pose(shape, frame_shape):
    # Mô hình 3D của khuôn mặt (chung)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Chóp mũi (Nose tip - 30)
        (0.0, -330.0, -65.0),        # Cằm (Chin - 8)
        (-225.0, 170.0, -135.0),     # Góc mắt trái (Left eye left corner - 36)
        (225.0, 170.0, -135.0),      # Góc mắt phải (Right eye right corner - 45)
        (-150.0, -150.0, -125.0),    # Khóe miệng trái (Left Mouth corner - 48)
        (150.0, -150.0, -125.0)      # Khóe miệng phải (Right Mouth corner - 54)
    ])
    
    # Lấy các điểm 2D tương ứng từ Dlib
    image_points = np.array([
        shape[30],     # Chóp mũi
        shape[8],      # Cằm
        shape[36],     # Góc mắt trái
        shape[45],     # Góc mắt phải
        shape[48],     # Khóe miệng trái
        shape[54]      # Khóe miệng phải
    ], dtype="double")
    
    # Thông số camera (giả định)
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    # Giả sử không có biến dạng ống kính
    dist_coeffs = np.zeros((4, 1)) 
    
    # Sử dụng cv2.solvePnP
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Chuyển đổi vector xoay sang Euler angles
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
    
    # Lấy góc Pitch, Yaw, Roll (đơn vị: độ)
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    
    return pitch, yaw, roll

# --- KHỞI TẠO CÁC BIẾN ---
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# --- CHUẨN BỊ GHI FILE CSV ---
# Đặt tên file video đầu vào và nhãn cho video đó
# 0 = Tỉnh táo, 1 = Buồn ngủ
VIDEO_SOURCE = "video_buon_ngu.mp4" 
LABEL_TO_WRITE = 1
# Khi chạy xong, bạn đổi thành:
# VIDEO_SOURCE = "video_buon_ngu.mp4" 
# LABEL_TO_WRITE = 1

CSV_FILE = 'drowsiness_data.csv'

# Mở file CSV để ghi
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    # Ghi tiêu đề (chỉ ghi một lần khi bắt đầu)
    # f.seek(0, 2) # di chuyển đến cuối file
    # if f.tell() == 0: # nếu file rỗng
    #     writer.writerow(['ear', 'mar', 'pitch', 'yaw', 'roll', 'label'])
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_shape = frame.shape
        
        subjects = detect(gray, 0)
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape_np = face_utils.shape_to_np(shape)
            
            # 1. Tính EAR
            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # 2. Tính MAR
            mouth = shape_np[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            
            # 3. Tính Head Pose
            pitch, yaw, roll = get_head_pose(shape_np, frame_shape)
            
            # In ra để kiểm tra
            print(f"EAR: {ear:.2f}, MAR: {mar:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Label: {LABEL_TO_WRITE}")
            
            # 4. Ghi dữ liệu vào file CSV
            writer.writerow([ear, mar, pitch, yaw, roll, LABEL_TO_WRITE])
            
            # (Phần code vẽ vời có thể thêm vào đây để xem)
            
        # cv2.imshow("Frame", frame) # Bỏ comment nếu muốn xem video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"Đã trích xuất xong dữ liệu từ {VIDEO_SOURCE}!")