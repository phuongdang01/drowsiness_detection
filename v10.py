
import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================

# EAR Thresholds (Adjustable)
# If AVG EAR < THRESHOLD for n consecutive frames -> Drowsiness
EAR_THRESH = 0.25 

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark Indices for EAR (Standard 6-point)
# Format: [P1(Left), P2(Top1), P3(Top2), P4(Right), P5(Bottom2), P6(Bottom1)]
# Note: MediaPipe indices are slightly different than Dlib 68 points.
# We pick specific points that map well to the eye corners and eyelids.

# Left Eye
# P1: 33 (Left Corner)
# P2: 160 (Top-Outer)
# P3: 158 (Top-Inner)
# P4: 133 (Right Corner - Inner Canthus)
# P5: 153 (Bottom-Inner)
# P6: 144 (Bottom-Outer)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

# Right Eye
# P1: 362 (Left Corner - Inner Canthus)
# P2: 385 (Top-Inner)
# P3: 387 (Top-Outer)
# P4: 263 (Right Corner)
# P5: 373 (Bottom-Outer)
# P6: 380 (Bottom-Inner)
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# For drawing just the eye countour (optional, helps visualization)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_ear(landmarks, indices, img_w, img_h):
    """
    Calculates Eye Aspect Ratio (EAR) for a single eye.
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # Get coordinates
    # indices order: [P1, P2, P3, P4, P5, P6]
    # P1: Left, P4: Right (Horizontal)
    # P2, P3: Top
    # P6, P5: Bottom
    
    # Extract points from landmarks
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x * img_w, lm.y * img_h]))
    
    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    h_dist = np.linalg.norm(p1 - p4)

    # EAR Formula
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

def draw_eye_landmarks(frame, landmarks, indices, img_w, img_h, color=(0, 255, 0)):
    """Draws the 6 EAR keypoints on the eye."""
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(frame, (x, y), 2, color, -1)

# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            h, w, _ = frame.shape
            
            # Convert to RGB
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Draw
            frame.flags.writeable = True
            
            # Default values
            ear_left = 0
            ear_right = 0
            avg_ear = 0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    # 1. Calculate EAR
                    ear_left = calculate_ear(lm, LEFT_EYE_IDXS, w, h)
                    ear_right = calculate_ear(lm, RIGHT_EYE_IDXS, w, h)
                    avg_ear = (ear_left + ear_right) / 2.0

                    # 2. Visualize Points
                    draw_eye_landmarks(frame, lm, LEFT_EYE_IDXS, w, h, (0, 255, 0))
                    draw_eye_landmarks(frame, lm, RIGHT_EYE_IDXS, w, h, (0, 255, 0))

                    # 3. Visual Feedback (Text)
                    color = (0, 255, 0)
                    status = "Open"
                    
                    if avg_ear < EAR_THRESH:
                        color = (0, 0, 255) # Red if closed
                        status = "Closed / Drowsy"
                    
                    cv2.putText(frame, f"Left EAR: {ear_left:.2f}", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Right EAR: {ear_right:.2f}", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"AVG EAR: {avg_ear:.2f}", (30, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame, f"Status: {status}", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('EAR Calculation - Drowsiness Detection', frame)
            
            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
