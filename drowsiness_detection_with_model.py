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

# ====================== MODEL DEFINITION ======================
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Closed, Open
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

# ====================== LOAD MODEL ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eye_model = EyeClassifier().to(device)
eye_model.load_state_dict(torch.load('eye_model.pth', map_location=device))
eye_model.eval()

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
MOUTH = [61, 291, 13, 14]

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

def predict_eye_state(eye_roi):
    """Predict if eye is open or closed using the trained model"""
    if eye_roi is None or eye_roi.size == 0:
        return 0, 0.0  # default to open
    
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB))
        
        # Transform
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = eye_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item()
        
        return predicted, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, 0.0

def play_alert_loop(stop_event):
    """Play alert sound in loop"""
    while not stop_event.is_set():
        try:
            playsound("alert.wav")
        except:
            print("Alert sound not found!")
            time.sleep(0.5)

# ====================== THRESHOLDS ======================
EAR_THRESH = 0.22
EAR_FRAMES = 15  # ~0.5 seconds at 30 fps
MAR_THRESH = 0.6
MAR_FRAMES = 15
CONFIDENCE_THRESH = 0.7  # For model prediction

# ====================== STATE VARIABLES ======================
eye_counter = 0
yawn_counter = 0
stop_event = threading.Event()
alert_thread = None

# ====================== MAIN LOOP ======================
# Use video from Video Database folder
video_path = r"deokinhthieusang.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

print(f"🚗 Drowsiness Detection System Started")
print(f"   Testing with video: {video_path}")
print("   Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    is_drowsy = False
    status_text = "Alert"
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        def get_xy(i):
            return (int(lm[i].x * w), int(lm[i].y * h))

        # ========== EYE DETECTION (Using Model) ==========
        left_eye_pts = [get_xy(i) for i in LEFT_EYE]
        right_eye_pts = [get_xy(i) for i in RIGHT_EYE]
        
        # Extract eye regions
        left_eye_roi, left_bbox = extract_eye_region(frame, left_eye_pts)
        right_eye_roi, right_bbox = extract_eye_region(frame, right_eye_pts)
        
        # Predict eye state
        left_pred, left_conf = predict_eye_state(left_eye_roi)
        right_pred, right_conf = predict_eye_state(right_eye_roi)
        
        # Calculate EAR as well (for display)
        leftEAR = eye_aspect_ratio(left_eye_pts)
        rightEAR = eye_aspect_ratio(right_eye_pts)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Both eyes closed with high confidence
        if left_pred == 0 and right_pred == 0 and \
           left_conf > CONFIDENCE_THRESH and right_conf > CONFIDENCE_THRESH:
            eye_counter += 1
            if eye_counter >= EAR_FRAMES:
                is_drowsy = True
                status_text = "DROWSY - EYES CLOSED!"
        else:
            eye_counter = 0
        
        # Draw eye boxes
        cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), 
                     (left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]), 
                     (0, 255, 0) if left_pred == 1 else (0, 0, 255), 2)
        cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), 
                     (right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]), 
                     (0, 255, 0) if right_pred == 1 else (0, 0, 255), 2)
        
        # ========== YAWN DETECTION (Using MAR) ==========
        mouth_pts = [get_xy(i) for i in MOUTH]
        mar = mouth_aspect_ratio(mouth_pts)
        
        if mar > MAR_THRESH:
            yawn_counter += 1
            if yawn_counter >= MAR_FRAMES:
                is_drowsy = True
                status_text = "DROWSY - YAWNING!"
        else:
            yawn_counter = 0
        
        # Draw mouth box
        mouth_rect = cv2.boundingRect(np.array(mouth_pts))
        cv2.rectangle(frame, (mouth_rect[0], mouth_rect[1]), 
                     (mouth_rect[0]+mouth_rect[2], mouth_rect[1]+mouth_rect[3]), 
                     (0, 0, 255) if mar > MAR_THRESH else (0, 255, 0), 2)
        
        # ========== DISPLAY INFO ==========
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"L-Eye: {'CLOSED' if left_pred == 0 else 'OPEN'} ({left_conf:.2f})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"R-Eye: {'CLOSED' if right_pred == 0 else 'OPEN'} ({right_conf:.2f})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # ========== DROWSINESS ALERT ==========
    if is_drowsy:
        cv2.putText(frame, status_text, (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Start alert sound if not already playing
        if alert_thread is None or not alert_thread.is_alive():
            stop_event.clear()
            alert_thread = threading.Thread(target=play_alert_loop, args=(stop_event,))
            alert_thread.daemon = True
            alert_thread.start()
    else:
        cv2.putText(frame, "Status: ALERT", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Stop alert sound
        if alert_thread and alert_thread.is_alive():
            stop_event.set()
    
    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================== CLEANUP ======================
cap.release()
cv2.destroyAllWindows()
stop_event.set()
print("System stopped.")
