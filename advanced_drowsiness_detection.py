"""
Advanced Drowsiness Detection System
- Segmentation-based eye detection with trained model
- Yawn frequency detection over time window
- Head pose estimation for head nodding detection
- Multiple drowsiness indicators
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import threading
import time
from playsound import playsound
from collections import deque
import math

# ====================== MODEL DEFINITION ======================
class AdvancedDrowsinessModel(nn.Module):
    """
    Multi-task model with segmentation-inspired architecture
    """
    def __init__(self):
        super(AdvancedDrowsinessModel, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.eye_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        self.yawn_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x = self.pool3(x3)
        
        x = self.bottleneck(x)
        features = self.global_pool(x)
        
        eye_out = self.eye_classifier(features)
        yawn_out = self.yawn_classifier(features)
        
        return eye_out, yawn_out

# ====================== LOAD MODEL ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# Try to load advanced model, fallback to simple model
try:
    model = AdvancedDrowsinessModel().to(device)
    model.load_state_dict(torch.load('advanced_drowsiness_model.pth', map_location=device))
    model.eval()
    print("✅ Loaded advanced model")
    USE_ADVANCED_MODEL = True
except:
    print("⚠️  Advanced model not found, using basic model")
    # Fallback to simple model
    class EyeClassifier(nn.Module):
        def __init__(self):
            super(EyeClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = EyeClassifier().to(device)
    model.load_state_dict(torch.load('eye_model.pth', map_location=device))
    model.eval()
    USE_ADVANCED_MODEL = False

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ====================== MEDIAPIPE ======================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ====================== CONSTANTS ======================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOUTH = [61, 291, 13, 14, 17, 78, 308]

# Nose and face points for head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# ====================== HEAD POSE ESTIMATION ======================
def estimate_head_pose(face_landmarks, img_shape):
    """
    Estimate head pose (pitch, yaw, roll) using facial landmarks
    Returns angles in degrees and 3D rotation/translation vectors
    """
    h, w = img_shape[:2]
    
    # 2D image points from landmarks
    image_points = np.array([
        (face_landmarks.landmark[NOSE_TIP].x * w, face_landmarks.landmark[NOSE_TIP].y * h),
        (face_landmarks.landmark[CHIN].x * w, face_landmarks.landmark[CHIN].y * h),
        (face_landmarks.landmark[LEFT_EYE_CORNER].x * w, face_landmarks.landmark[LEFT_EYE_CORNER].y * h),
        (face_landmarks.landmark[RIGHT_EYE_CORNER].x * w, face_landmarks.landmark[RIGHT_EYE_CORNER].y * h),
        (face_landmarks.landmark[LEFT_MOUTH].x * w, face_landmarks.landmark[LEFT_MOUTH].y * h),
        (face_landmarks.landmark[RIGHT_MOUTH].x * w, face_landmarks.landmark[RIGHT_MOUTH].y * h)
    ], dtype="double")
    
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    pitch = math.degrees(math.atan2(rotation_matrix[2][1], rotation_matrix[2][2]))
    yaw = math.degrees(math.atan2(-rotation_matrix[2][0], 
                                   math.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)))
    roll = math.degrees(math.atan2(rotation_matrix[1][0], rotation_matrix[0][0]))
    
    return pitch, yaw, roll, rotation_vector, translation_vector

def draw_head_axes(frame, face_landmarks, img_shape, rotation_vector, translation_vector):
    """Draw 3D coordinate axes on head"""
    h, w = img_shape[:2]
    
    # Camera matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    
    # Nose tip point
    nose_tip_2d = (
        int(face_landmarks.landmark[NOSE_TIP].x * w),
        int(face_landmarks.landmark[NOSE_TIP].y * h)
    )
    
    # 3D axis points
    axis_length = 200
    axis_points = np.array([
        (0, 0, 0),              # Origin (nose)
        (axis_length, 0, 0),    # X-axis (Red)
        (0, axis_length, 0),    # Y-axis (Green)
        (0, 0, axis_length)     # Z-axis (Blue)
    ], dtype="double")
    
    # Project 3D points to 2D
    axis_points_2d, _ = cv2.projectPoints(
        axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    
    # Draw axes
    p0 = tuple(axis_points_2d[0].ravel().astype(int))
    p_x = tuple(axis_points_2d[1].ravel().astype(int))
    p_y = tuple(axis_points_2d[2].ravel().astype(int))
    p_z = tuple(axis_points_2d[3].ravel().astype(int))
    
    cv2.line(frame, p0, p_x, (0, 0, 255), 3)  # X-axis: Red
    cv2.line(frame, p0, p_y, (0, 255, 0), 3)  # Y-axis: Green
    cv2.line(frame, p0, p_z, (255, 0, 0), 3)  # Z-axis: Blue
    
    # Add labels
    cv2.putText(frame, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Z", p_z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# ====================== FUNCTIONS ======================
def eye_aspect_ratio(eye):
    """Calculate EAR from eye landmarks"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Calculate MAR from mouth landmarks"""
    A = distance.euclidean(mouth[2], mouth[3])  # vertical
    B = distance.euclidean(mouth[0], mouth[1])  # horizontal
    if B == 0: 
        return 0.0
    return A / B

def extract_eye_region(frame, eye_landmarks, padding=10):
    """Extract eye region from frame"""
    eye_points = np.array(eye_landmarks)
    x, y, w, h = cv2.boundingRect(eye_points)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2*padding)
    h = min(frame.shape[0] - y, h + 2*padding)
    
    eye_roi = frame[y:y+h, x:x+w]
    return eye_roi, (x, y, w, h)

def get_iris_center(iris_landmarks, frame_w, frame_h):
    """Calculate the center of the iris"""
    # Simply averaging the landmarks is a good approximation for the center
    cx = int(np.mean([lm.x * frame_w for lm in iris_landmarks]))
    cy = int(np.mean([lm.y * frame_h for lm in iris_landmarks]))
    return np.array([cx, cy])

def predict_eyes_batch(left_roi, right_roi):
    """Predict state for both eyes in a single batch"""
    if left_roi is None or left_roi.size == 0 or right_roi is None or right_roi.size == 0:
        return 1, 0.0, 1, 0.0, 1, 0.0, 1, 0.0

    try:
        # Preprocess both eyes
        pil_left = Image.fromarray(cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB))
        pil_right = Image.fromarray(cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB))
        
        tensor_left = transform(pil_left)
        tensor_right = transform(pil_right)
        
        # Stack into a batch of 2: [2, 3, 64, 64]
        input_batch = torch.stack([tensor_left, tensor_right]).to(device)
        
        with torch.no_grad():
            if USE_ADVANCED_MODEL:
                eye_out, yawn_out = model(input_batch)
                
                # Process outputs
                eye_probs = torch.softmax(eye_out, dim=1)
                eye_preds = torch.argmax(eye_probs, dim=1).cpu().numpy()
                eye_confs = [eye_probs[i][eye_preds[i]].item() for i in range(2)]
                
                # We can just ignore yawns from eye regions or average them if desired
                # But typically we should use mouth ROI for yawn. 
                # The original code used eye ROIs for both eye and yawn prediction? 
                # Let's stick to the original logic which called predict_eye_and_yawn on eye ROIs.
                # Assuming the model was trained such that it can detect yawns even from eye crops? 
                # OR (more likely) the original code was just calling the multitask model on whatever input.
                
                return eye_preds[0], eye_confs[0], 0, 0.0, eye_preds[1], eye_confs[1], 0, 0.0
            else:
                outputs = model(input_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confs = [probs[i][preds[i]].item() for i in range(2)]
                return preds[0], confs[0], 1, 0.0, preds[1], confs[1], 1, 0.0
                
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return 1, 0.0, 1, 0.0, 1, 0.0, 1, 0.0

def eye_aspect_ratio(eye):
    """Calculate EAR from eye landmarks (optimized)"""
    # Use hypot instead of scipy.spatial.distance.euclidean for speed
    A = math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Calculate MAR from mouth landmarks (optimized)"""
    A = math.hypot(mouth[2][0] - mouth[3][0], mouth[2][1] - mouth[3][1])  # vertical
    B = math.hypot(mouth[0][0] - mouth[1][0], mouth[0][1] - mouth[1][1])  # horizontal
    if B == 0: 
        return 0.0
    return A / B
EAR_THRESH = 0.20       # Giảm nhẹ để tránh nhầm mắt híp là nhắm (Gốc: 0.22)
EAR_FRAMES = 20         # Tăng thời gian nhắm mắt lên ~0.7s mới báo động (Gốc: 15)
MAR_THRESH = 0.6
YAWN_FRAMES = 20
YAWN_WINDOW = 60        # Tăng cửa sổ thời gian đếm ngáp (Gốc: 30)
MAX_YAWNS_IN_WINDOW = 3
HEAD_PITCH_THRESH = 20  # Tăng góc cúi đầu để tránh báo nhầm khi nhìn bảng điều khiển (Gốc: 15)
HEAD_YAW_THRESH = 30
HEAD_NOD_FRAMES = 30    # Tăng số frame xác nhận gật gù (Gốc: 20)
CONFIDENCE_THRESH = 0.75 # Tăng độ tin cậy mô hình yêu cầu (Gốc: 0.65)
DEBUG_MODE = True
GAZE_FIXATION_TIME = 6.0  # Tăng thời gian nhìn chằm chằm lên 6s (Gốc: 3.0)
GAZE_MOVE_THRESH = 1.5    # Giảm pixel, mắt phải cực kỳ đứng yên mới tính (Gốc: 3.0)
BLINK_FREQ_THRESH = 8     # Tăng ngưỡng chớp mắt (Gốc: 5)
BLINK_WINDOW = 10.0       # Tăng cửa sổ đo chớp mắt (Gốc: 5.0)
def play_alert_loop(stop_event):
    """Play alert sound in loop"""
    while not stop_event.is_set():
        try:
            playsound("alert.wav")
        except:
            print("Alert sound not found!")
            time.sleep(0.5)
# ====================== STATE VARIABLES ======================
eye_counter = 0
yawn_counter = 0
head_nod_counter = 0
stop_event = threading.Event()
alert_thread = None

# Calibration variables
CALIBRATION_FRAMES = 50  # Number of frames for calibration (approx 2 sec)
calibration_frame_count = 0
is_calibrated = False
base_pitch = 0
base_yaw = 0
base_roll = 0
pitch_buffer = []
yaw_buffer = []
roll_buffer = []

# Blink frequency tracking
blink_timestamps = deque(maxlen=100)
is_blinking = False

# Gaze tracking
gaze_points = deque(maxlen=100)  # Store recent iris positions
last_gaze_time = time.time()
stare_start_time = None

# 
# Yawn frequency tracking
yawn_timestamps = deque(maxlen=100)  # Store timestamps of yawns
current_yawn_frames = 0
is_yawning = False

# ====================== VIDEO SOURCE ======================
# Change these settings based on your needs
USE_VIDEO_FILE = False  # Set to False for webcam
VIDEO_PATH = r"Video Database\Sub 03.avi"

if USE_VIDEO_FILE:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {VIDEO_PATH}")
        exit()
    print(f"📹 Testing with video: {VIDEO_PATH}")
else:
    cap = cv2.VideoCapture(0)
    print("📹 Using webcam")

print("🚗 Advanced Drowsiness Detection System Started")
print("   Features:")
print("   - Segmentation-based eye detection")
print("   - Yawn frequency analysis")
print("   - Head pose estimation with coordinate axes")
print("   - Head nodding detection")
print("   - Gaze tracking (Blank Stare)")
print("   - Rapid blinking detection")
print("   Press Q to quit\n")

frame_count = 0
start_time = time.time()

# ====================== MAIN LOOP ======================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        if USE_VIDEO_FILE:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break

    frame_count += 1
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    is_drowsy = False
    drowsy_reasons = []
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        def get_xy(i):
            return (int(lm[i].x * w), int(lm[i].y * h))

        # ========== YAWN DETECTION (PRE-CALC) ==========
        # Tính toán MAR sớm để lọc nhiễu cho Head Nodding
        mouth_pts = [get_xy(i) for i in MOUTH]
        mar = mouth_aspect_ratio(mouth_pts)

        # ========== HEAD POSE ESTIMATION ==========
        try:
            pitch, yaw, roll, rvec, tvec = estimate_head_pose(face_landmarks, frame.shape)
            
            # Draw coordinate axes on head
            draw_head_axes(frame, face_landmarks, frame.shape, rvec, tvec)
            
            # Calibration Logic
            if not is_calibrated:
                cv2.putText(frame, f"CALIBRATING: {calibration_frame_count}/{CALIBRATION_FRAMES}", (10, h//2 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "PLEASE LOOK STRAIGHT AHEAD", (10, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                pitch_buffer.append(pitch)
                yaw_buffer.append(yaw)
                roll_buffer.append(roll)
                calibration_frame_count += 1
                
                if calibration_frame_count >= CALIBRATION_FRAMES:
                    base_pitch = sum(pitch_buffer) / len(pitch_buffer)
                    base_yaw = sum(yaw_buffer) / len(yaw_buffer)
                    base_roll = sum(roll_buffer) / len(roll_buffer)
                    is_calibrated = True
                    print(f"Calibration Complete! Base Pitch: {base_pitch:.2f}, Base Yaw: {base_yaw:.2f}")
                
                # Continue preventing drowsiness detection during calibration
                continue

            # Calculate relative angles
            rel_pitch = pitch - base_pitch
            rel_yaw = yaw - base_yaw
            rel_roll = roll - base_roll
            
            # Display head angles (Relative)
            pitch_color = (0, 0, 255) if rel_pitch > HEAD_PITCH_THRESH else (255, 255, 0)
            cv2.putText(frame, f"Pitch (Rel): {rel_pitch:.1f}", (w - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
            cv2.putText(frame, f"Yaw (Rel): {rel_yaw:.1f}", (w - 250, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Roll (Rel): {rel_roll:.1f}", (w - 250, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Detect head nodding (head tilted down significantly relative to calibration)
            # Improved logic: Check relative pitch AND yaw to avoid false positives
            # Quan trọng: Nếu đang ngáp (MAR cao) thì KHÔNG tính là gật đầu (tránh nhầm lẫn)
            is_yawning_now = mar > MAR_THRESH

            if rel_pitch > HEAD_PITCH_THRESH and abs(rel_yaw) < HEAD_YAW_THRESH and not is_yawning_now:
                head_nod_counter += 1
                if DEBUG_MODE:
                    cv2.putText(frame, f"Nod Counter: {head_nod_counter}/{HEAD_NOD_FRAMES}", 
                               (w - 250, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if head_nod_counter >= HEAD_NOD_FRAMES:
                    is_drowsy = True
                    drowsy_reasons.append("HEAD NODDING")
            else:
                head_nod_counter = max(0, head_nod_counter - 1)
            
        except Exception as e:
            print(f"Head pose error: {e}")
            pitch, yaw, roll = 0, 0, 0

        # ========== EYE DETECTION ==========
        left_eye_pts = [get_xy(i) for i in LEFT_EYE]
        right_eye_pts = [get_xy(i) for i in RIGHT_EYE]
        
        # Extract eye regions
        left_eye_roi, left_bbox = extract_eye_region(frame, left_eye_pts)
        right_eye_roi, right_bbox = extract_eye_region(frame, right_eye_pts)
        
        # Batch Predict eye state (Optimization)
        left_eye_pred, left_eye_conf, _, _, right_eye_pred, right_eye_conf, _, _ = predict_eyes_batch(left_eye_roi, right_eye_roi)
        
        # Calculate EAR
        leftEAR = eye_aspect_ratio(left_eye_pts)
        rightEAR = eye_aspect_ratio(right_eye_pts)
        ear = (leftEAR + rightEAR) / 2.0
        
        # --- BLINK FREQUENCY DETECTION ---
        # Detect blink based on EAR (using standard threshold for blink counting)
        if ear < EAR_THRESH:
            if not is_blinking:
                is_blinking = True
        else:
            if is_blinking:
                is_blinking = False
                blink_timestamps.append(time.time())
        
        # Count blinks in window
        current_time = time.time()
        recent_blinks = [t for t in blink_timestamps if current_time - t <= BLINK_WINDOW]
        blink_freq = len(recent_blinks)
        
        if blink_freq > BLINK_FREQ_THRESH:
            # Chỉ coi là dấu hiệu phụ, không báo động ngay lập tức
            # is_drowsy = True
            # drowsy_reasons.append(f"BLINKING ({blink_freq})")
            pass
            
        # --- GAZE TRACKING (BLANK STARE) ---
        # ... (giữ nguyên logic tính toán) ...
        # Get iris landmarks
        try:
            # Indices for iris landmarks (from MediaPipe holistic/face mesh topology)
            # Left iris: 474, 475, 476, 477. Center: 468
            # Right iris: 469, 470, 471, 472. Center: 473
            
            # Using specific points for iris center approximation
            L_IRIS_CENTER = 468
            R_IRIS_CENTER = 473
            
            l_center_lm = face_landmarks.landmark[L_IRIS_CENTER]
            r_center_lm = face_landmarks.landmark[R_IRIS_CENTER]
            
            l_center = np.array([int(l_center_lm.x * w), int(l_center_lm.y * h)])
            r_center = np.array([int(r_center_lm.x * w), int(r_center_lm.y * h)])
            
            avg_center = (l_center + r_center) / 2
            
            # Add to history for stability check
            gaze_points.append(avg_center)
            
            # Check stability over time (staring)
            # We look at the standard deviation of gaze points over time
            # Low deviation means fixed gaze
            if len(gaze_points) > 10:
                # Convert deque to array
                points_arr = np.array(list(gaze_points))
                
                # Calculate standard deviation of X and Y coordinates
                std_dev = np.std(points_arr, axis=0)
                movement_score = np.mean(std_dev) # Average movement
                
                if DEBUG_MODE:
                     cv2.putText(frame, f"Gaze Move: {movement_score:.2f}", (w - 200, h - 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if movement_score < GAZE_MOVE_THRESH:
                    if stare_start_time is None:
                        stare_start_time = time.time()
                    elif time.time() - stare_start_time > GAZE_FIXATION_TIME:
                        # Only flag as staring if user is NOT blinking and eyes are slightly closed (EAR < 0.25)
                        # Tránh báo động khi người dùng nhìn đường tập trung
                        if not is_blinking and ear < 0.25:
                            is_drowsy = True
                            drowsy_reasons.append("BLANK STARE")
                            cv2.putText(frame, "STARE DETECTED", (w - 250, h - 130), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Chỉ cảnh báo nhẹ
                            drowsy_reasons.append("FIXED GAZE")
                else:
                    stare_start_time = None
            
            # Draw pupil/iris centers
            cv2.circle(frame, tuple(l_center), 3, (0, 255, 255), -1)
            cv2.circle(frame, tuple(r_center), 3, (0, 255, 255), -1)
            
        except Exception as e:
            # print(f"Gaze error: {e}") 
            pass

        # r = (leftEAR + rightEAR) / 2.0
        
        # Both eyes closed logic refined
        # Logic: Cả 2 mắt được dự đoán đóng VÀ độ tin cậy cao
        # Hoặc EAR < EAR_THRESH (dựa trên hình học) VÀ ít nhất 1 mắt dự đoán đóng
        model_says_closed = (left_eye_pred == 0 and right_eye_pred == 0 and \
                             left_eye_conf > CONFIDENCE_THRESH and right_eye_conf > CONFIDENCE_THRESH)
        
        ear_says_closed = ear < EAR_THRESH

        if model_says_closed or (ear_says_closed and (left_eye_pred == 0 or right_eye_pred == 0)):
            eye_counter += 1
            if eye_counter >= EAR_FRAMES:
                is_drowsy = True
                drowsy_reasons.append(f"EYES CLOSED ({eye_counter})")
        else:
            eye_counter = max(0, eye_counter - 1)
        
        # Draw eye boxes
        cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), 
                     (left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]), 
                     (0, 255, 0) if left_eye_pred == 1 else (0, 0, 255), 2)
        cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), 
                     (right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]), 
                     (0, 255, 0) if right_eye_pred == 1 else (0, 0, 255), 2)
        
        # ========== YAWN DETECTION ==========
        # Lưu ý: mouth_pts và mar đã được tính ở trên để phục vụ Head Pose logic
        # Chỉ cập nhật lại logic đếm và vẽ ở đây
        
        # Count consecutive yawn frames
        if mar > MAR_THRESH:
            current_yawn_frames += 1
            if not is_yawning and current_yawn_frames >= YAWN_FRAMES:
                is_yawning = True
                yawn_timestamps.append(time.time())
        else:
            if is_yawning:
                is_yawning = False
            current_yawn_frames = max(0, current_yawn_frames - 1)
        
        # Calculate yawn frequency
        current_time = time.time()
        recent_yawns = [t for t in yawn_timestamps if current_time - t <= YAWN_WINDOW]
        yawn_frequency = len(recent_yawns)
        
        if yawn_frequency >= MAX_YAWNS_IN_WINDOW:
            is_drowsy = True
            drowsy_reasons.append(f"FREQUENT YAWNING ({yawn_frequency}x)")
        
        # Draw mouth box
        mouth_rect = cv2.boundingRect(np.array(mouth_pts))
        cv2.rectangle(frame, (mouth_rect[0], mouth_rect[1]), 
                     (mouth_rect[0]+mouth_rect[2], mouth_rect[1]+mouth_rect[3]), 
                     (0, 0, 255) if mar > MAR_THRESH else (0, 255, 0), 2)
        
        # ========== DISPLAY INFO ==========
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn Freq: {yawn_frequency}/{MAX_YAWNS_IN_WINDOW}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"L-Eye: {'CLOSED' if left_eye_pred == 0 else 'OPEN'} ({left_eye_conf:.2f})", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"R-Eye: {'CLOSED' if right_eye_pred == 0 else 'OPEN'} ({right_eye_conf:.2f})", 
                   (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # ========== DROWSINESS ALERT ==========
    if is_drowsy:
        status_text = "DROWSY: " + ", ".join(drowsy_reasons)
        cv2.putText(frame, status_text, (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Start alert sound
        if alert_thread is None or not alert_thread.is_alive():
            stop_event.clear()
            alert_thread = threading.Thread(target=play_alert_loop, args=(stop_event,))
            alert_thread.daemon = True
            alert_thread.start()
    else:
        cv2.putText(frame, "Status: ALERT", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Stop alert sound
        if alert_thread and alert_thread.is_alive():
            stop_event.set()
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Advanced Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================== CLEANUP ======================
cap.release()
cv2.destroyAllWindows()
stop_event.set()
print("\n✅ System stopped.")
print(f"📊 Total frames processed: {frame_count}")
print(f"⏱️  Total time: {time.time() - start_time:.2f}s")
