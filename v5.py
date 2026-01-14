from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    """
    Tính toán tỷ lệ khung mắt (Eye Aspect Ratio - EAR)
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # Tính toán khoảng cách euclidean giữa hai cặp
    # tọa độ (x, y) dọc của mắt
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Tính toán khoảng cách euclidean giữa các
    # tọa độ (x, y) ngang của mắt
    C = dist.euclidean(eye[0], eye[3])

    # Tính toán EAR
    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mouth):
    """
    Tính toán tỷ lệ khung miệng (Mouth Aspect Ratio - MAR)
    MAR = |p2 - p8| / |p1 - p5| (đơn giản hóa)
    Chúng ta sẽ sử dụng một phép đo đơn giản hơn là khoảng cách dọc
    giữa môi trên và môi dưới
    """
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59 (chỉ số 68-point)
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55 (ngang)

    # Tính MAR
    mar = (A + B) / (2.0 * C)
    return mar

# --- KHỞI TẠO CÁC HẰNG SỐ VÀ THAM SỐ ---

# Đường dẫn đến tệp mô hình dự đoán đặc điểm
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Định nghĩa 2 hằng số:
# 1. Ngưỡng EAR: Dưới mức này coi là "nhắm mắt"
EYE_AR_THRESH = 0.27
# 2. Số khung hình liên tiếp mắt phải nhắm để kích hoạt cảnh báo
EYE_AR_CONSEC_FRAMES = 25 # Khoảng 1 giây

# Ngưỡng MAR: Trên mức này coi là "ngáp"
MOUTH_AR_THRESH = 0.65
# Số khung hình liên tiếp ngáp để kích hoạt
MOUTH_AR_CONSEC_FRAMES = 30

# Khởi tạo bộ đếm khung hình và trạng thái cảnh báo
EYE_COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False

# --- KHỞI TẠO CÁC CÔNG CỤ CỦA DLIB ---

print("[INFO] Đang tải bộ dự đoán đặc điểm khuôn mặt...")
# 1. Bộ phát hiện khuôn mặt (dựa trên HOG)
detector = dlib.get_frontal_face_detector()

# 2. Bộ dự đoán đặc điểm (cần tệp .dat)
try:
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except RuntimeError as e:
    print(f"[LỖI] Không thể tải tệp 'shape_predictor_68_face_landmarks.dat'.")
    print(f"Hãy tải về từ: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print(f"Và đặt nó vào cùng thư mục với script này.")
    sys.exit()
    
# Lấy chỉ số (indices) cho mắt trái, mắt phải và miệng
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# --- KHỞI ĐỘNG LUỒNG VIDEO ---

print("[INFO] Đang khởi động luồng video (webcam)...")
# Sử dụng VideoStream từ imutils để xử lý đa luồng, giúp webcam mượt hơn
vs = VideoStream(src=0).start()
# Chờ 1 giây để camera khởi động
time.sleep(1.0)

# --- VÒNG LẶP CHÍNH XỬ LÝ TỪNG KHUNG HÌNH ---

while True:
    # Đọc khung hình từ luồng video, lật ngang, và thay đổi kích thước
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # Lật ngang khung hình (như nhìn vào gương)
    frame = cv2.flip(frame, 1)
    
    # Chuyển sang ảnh xám (grayscale) để dlib xử lý nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh xám
    rects = detector(gray, 0)

    # Lặp qua từng khuôn mặt được phát hiện
    for rect in rects:
        # 1. Lấy tọa độ khuôn mặt (x, y, w, h)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # Vẽ một hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 2. Xác định các đặc điểm khuôn mặt (landmarks)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape) # Chuyển sang mảng NumPy

        # 3. Trích xuất tọa độ mắt và miệng
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # 4. Tính EAR và MAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # Lấy trung bình EAR của cả hai mắt
        ear = (leftEAR + rightEAR) / 2.0
        
        mar = mouth_aspect_ratio(mouth)

        # 5. Vẽ đường viền xung quanh mắt và miệng
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        # 6. Hiển thị giá trị EAR và MAR lên màn hình
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- XỬ LÝ LOGIC NGỦ GẬT VÀ NGÁP ---

        # 7. Kiểm tra logic nhắm mắt (EAR)
        if ear < EYE_AR_THRESH:
            EYE_COUNTER += 1

            # Nếu mắt nhắm đủ lâu
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("[CẢNH BÁO] Mắt nhắm quá lâu!")
                
                # Vẽ cảnh báo lên màn hình
                cv2.putText(frame, "!!! CANH BAO NGU GAT !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            EYE_COUNTER = 0
            ALARM_ON = False # Tắt cảnh báo nếu mắt đã mở lại

        # 8. Kiểm tra logic ngáp (MAR)
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1
            
            if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                 if not ALARM_ON:
                    ALARM_ON = True
                    print("[CẢNH BÁO] Phát hiện ngáp!")
                
                 cv2.putText(frame, "!!! CANH BAO NGAP !!!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            MOUTH_COUNTER = 0
            # (Chúng ta có thể muốn giữ ALARM_ON = False ở logic của Mắt)
            
    # Hiển thị khung hình kết quả
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Nhấn 'q' để thoát
    if key == ord("q"):
        break

# --- DỌN DẸP ---
print("[INFO] Đang dọn dẹp và thoát...")
cv2.destroyAllWindows()
vs.stop()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              