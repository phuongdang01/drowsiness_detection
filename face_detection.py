import cv2
import mediapipe as mp

# Khởi tạo các mô-đun Mediapipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh sang RGB (Mediapipe dùng RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Vẽ khung nhận diện khuôn mặt
        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)

        cv2.imshow('Face Detection', frame)

        # Nhấn ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
