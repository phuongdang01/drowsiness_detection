import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import joblib # Thư viện mới
import beepy as beep

# --- TẢI MODEL VÀ SCALER ĐÃ TRAIN ---
model = joblib.load('drowsiness_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- IMPORT CÁC HÀM TÍNH TOÁN (TỪ SCRIPT 1) ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

def get_head_pose(shape, frame_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    (success, rvec, tvec) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    (rmat, jac) = cv2.Rodrigues(rvec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2] # pitch, yaw, roll

frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag = 0
alarm_on = False # <-- 2. KHỞI TẠO CỜ BÁO ĐỘNG

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
        
        # 1. Tính toán TẤT CẢ các đặc trưng
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mouth = shape_np[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        pitch, yaw, roll = get_head_pose(shape_np, frame_shape)

        # 2. ĐÓNG GÓI DỮ LIỆU ĐỂ DỰ ĐOÁN
        # Tạo một mảng 1D chứa các đặc trưng
        current_features = np.array([[ear, mar, pitch, yaw, roll]])

        # 3. CHUẨN HÓA DỮ LIỆU
        # (Phải dùng scaler đã fit ở script 2)
        current_features_scaled = scaler.transform(current_features)

        # 4. DỰ ĐOÁN
        prediction = model.predict(current_features_scaled) # Sẽ trả về [0] hoặc [1]
        # (Vẽ viền mắt/miệng để debug)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 5. LOGIC CẢNH BÁO MỚI (Dựa trên model)
        if prediction[0] == 1: # Nếu model dự đoán là "Buồn ngủ"
            flag += 1
            print(f"Flag: {flag}")
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # --- PHẦN ÂM THANH MỚI ---
                if not alarm_on:
                    beep.beep(4) # Phát âm thanh (số 4 là 'error')
                    alarm_on = True # Đặt cờ để không phát lặp lại
                # --- KẾT THÚC PHẦN ÂM THANH ---
        else:
            flag = 0 # Reset cờ nếu model nói "Tỉnh táo"
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
cap.release()