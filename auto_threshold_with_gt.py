"""
auto_threshold_with_gt.py

Chạy tự động nhiều ngưỡng EAR/MAR/NOD trên 1 video,
so sánh với ground truth timestamps, tính precision/recall/F1,
xuất results.csv, lưu detections per-threshold và vẽ biểu đồ top F1.

Yêu cầu: mediapipe, opencv-python, numpy, scipy, matplotlib
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import csv
import math
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "video_buon_ngu.mp4"
GROUND_TRUTH_TXT = "ground_truth.txt"
OUTPUT_CSV = "results.csv"
OUTPUT_DIR = "events"                  # lưu detections theo ngưỡng
PLOT_FILE = "f1_top15.png"
TIME_TOLERANCE = 1.5  # giây: khoảng cho phép khi so khớp detection <-> GT

# Các danh sách ngưỡng (thử tổ hợp)
EAR_THRESH_LIST = [0.20, 0.22, 0.24, 0.26]
EAR_FRAMES_LIST = [10, 12, 15]
MAR_THRESH_LIST = [0.48, 0.52, 0.58]
MAR_FRAMES_LIST = [8, 10, 12]
NOD_PITCH_LIST = [18.0, 22.0, 28.0]  # degrees (hoặc thử tăng)
NOD_FRAMES_LIST = [8, 10, 12]

# smoothing pitch frames
PITCH_SMOOTH_WINDOW = 5

# -------------------------
# Helper functions
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_PTS = [61, 291, 13, 14]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[3])  # trên - dưới
    B = distance.euclidean(mouth[0], mouth[1])  # trái - phải
    if B == 0:
        return 0.0
    return A / B

def extract_landmark_xy(lm, idx, w, h):
    return (int(lm[idx].x * w), int(lm[idx].y * h))

# Từ danh sách detection timestamps (s) và ground truth (s) -> TP/FP/FN
def match_detections_to_gt(detections, ground_truth, tol=TIME_TOLERANCE):
    # detections, ground_truth: sorted lists of seconds
    detections = sorted(detections)
    ground_truth = sorted(ground_truth)
    matched_det = set()
    matched_gt = set()

    # greedy: for each GT find nearest detection within tol
    for i, gt in enumerate(ground_truth):
        for j, det in enumerate(detections):
            if j in matched_det:
                continue
            if abs(det - gt) <= tol:
                matched_det.add(j)
                matched_gt.add(i)
                break

    TP = len(matched_gt)
    FP = len(detections) - len(matched_det)
    FN = len(ground_truth) - len(matched_gt)
    return TP, FP, FN

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_ground_truth(path):
    if not os.path.exists(path):
        print(f"[WARN] Ground truth file {path} not found. Using empty list.")
        return []
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    out = []
    for L in lines:
        try:
            t = float(L.strip())
            out.append(t)
        except:
            pass
    return sorted(out)

# -------------------------
# Core: run one test config
# -------------------------
def run_single_config(video_path, ear_t, ear_f, mar_t, mar_f, nod_t, nod_f):
    """
    Trả về list detections (giây) khi event xảy ra (khi ANY of the three conditions
    vượt ngưỡng: ear_counter >= ear_f OR yawn_counter >= mar_f OR nod_counter >= nod_f)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    ear_counter = 0
    yawn_counter = 0
    nod_counter = 0
    pitch_history = []

    detections = []       # timestamps (s) của event
    last_detection_time = -999.0
    MIN_SEPARATION = 1.0  # giây: tránh ghi nhiều detection liên tiếp khi cùng 1 event

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame_idx += 1
        t_seconds = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            def xy(i): return extract_landmark_xy(lm, i, w, h)

            left_eye = [xy(i) for i in LEFT_EYE]
            right_eye = [xy(i) for i in RIGHT_EYE]
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            mouth_pts = [xy(i) for i in MOUTH_PTS]
            mar = mouth_aspect_ratio(mouth_pts)

            # head pitch estimation via nose-chin vector angle approximation
            nose = xy(1)
            chin = xy(199) if 199 < len(lm) else (xy(152) if 152 < len(lm) else (xy(2)))
            dx = chin[0] - nose[0]
            dy = chin[1] - nose[1]
            pitch = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

            # smooth pitch
            pitch_history.append(pitch)
            if len(pitch_history) > PITCH_SMOOTH_WINDOW:
                pitch_history.pop(0)
            smooth_pitch = sum(pitch_history) / len(pitch_history)

            # counters
            if ear < ear_t:
                ear_counter += 1
            else:
                ear_counter = 0

            if mar > mar_t:
                yawn_counter += 1
            else:
                yawn_counter = 0

            valid_head = (ear > ear_t) and (mar < mar_t)
            if valid_head and smooth_pitch > nod_t:
                nod_counter += 1
            else:
                nod_counter = 0

            # event decision
            if (ear_counter >= ear_f) or (yawn_counter >= mar_f) or (nod_counter >= nod_f):
                # consolidate events: chỉ ghi 1 event trong 1 khoảng MIN_SEPARATION giây
                if t_seconds - last_detection_time >= MIN_SEPARATION:
                    detections.append(round(t_seconds, 3))
                    last_detection_time = t_seconds

    cap.release()
    return detections

# -------------------------
# MAIN: chạy tất cả tổ hợp
# -------------------------
def main():
    safe_mkdir(OUTPUT_DIR)
    gt_list = read_ground_truth(GROUND_TRUTH_TXT)
    print(f"Ground truth loaded: {len(gt_list)} timestamps")

    # header CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["EAR_T", "EAR_F", "MAR_T", "MAR_F", "NOD_T", "NOD_F",
                         "Detections", "TP", "FP", "FN", "Precision", "Recall", "F1"])

        results_rows = []

        total_configs = (len(EAR_THRESH_LIST) * len(EAR_FRAMES_LIST) *
                         len(MAR_THRESH_LIST) * len(MAR_FRAMES_LIST) *
                         len(NOD_PITCH_LIST) * len(NOD_FRAMES_LIST))
        cur = 0

        for ear_t in EAR_THRESH_LIST:
            for ear_f in EAR_FRAMES_LIST:
                for mar_t in MAR_THRESH_LIST:
                    for mar_f in MAR_FRAMES_LIST:
                        for nod_t in NOD_PITCH_LIST:
                            for nod_f in NOD_FRAMES_LIST:
                                cur += 1
                                print(f"[{cur}/{total_configs}] Testing EAR={ear_t} f={ear_f}, "
                                      f"MAR={mar_t} f={mar_f}, NOD={nod_t} f={nod_f}")

                                detections = run_single_config(
                                    VIDEO_PATH, ear_t, ear_f, mar_t, mar_f, nod_t, nod_f
                                )

                                # match with ground truth
                                TP, FP, FN = match_detections_to_gt(detections, gt_list, tol=TIME_TOLERANCE)
                                prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                                rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                                # write row
                                writer.writerow([ear_t, ear_f, mar_t, mar_f, nod_t, nod_f,
                                                 len(detections), TP, FP, FN,
                                                 round(prec, 4), round(rec, 4), round(f1, 4)])

                                results_rows.append({
                                    "ear_t": ear_t, "ear_f": ear_f,
                                    "mar_t": mar_t, "mar_f": mar_f,
                                    "nod_t": nod_t, "nod_f": nod_f,
                                    "detections": detections,
                                    "TP": TP, "FP": FP, "FN": FN, "prec": prec, "rec": rec, "f1": f1
                                })

                                # save detections for this config for inspection
                                key = f"EAR{ear_t}_f{ear_f}_MAR{mar_t}_f{mar_f}_NOD{nod_t}_f{nod_f}"
                                fname = os.path.join(OUTPUT_DIR, f"{key}.csv")
                                with open(fname, "w", newline="", encoding="utf-8") as fd:
                                    cw = csv.writer(fd)
                                    cw.writerow(["detection_sec"])
                                    for d in detections:
                                        cw.writerow([d])

        # chọn ngưỡng tốt nhất theo F1
        best = sorted(results_rows, key=lambda x: x["f1"], reverse=True)[0]
        print("\n=== BEST THRESHOLD (by F1) ===")
        print(best)
        print(f"CSV results saved to: {os.path.abspath(OUTPUT_CSV)}")
        print(f"All per-config detection files saved to: {os.path.abspath(OUTPUT_DIR)}")

        # vẽ top 15 theo F1
        sorted_by_f1 = sorted(results_rows, key=lambda x: x["f1"], reverse=True)[:15]
        labels = []
        f1s = []
        for r in sorted_by_f1:
            label = f"EAR{r['ear_t']}_f{r['ear_f']}\nMAR{r['mar_t']}_f{r['mar_f']}\nNOD{r['nod_t']}_f{r['nod_f']}"
            labels.append(label)
            f1s.append(r["f1"])

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(labels))[::-1], f1s, align='center')
        plt.yticks(range(len(labels))[::-1], labels)
        plt.xlabel("F1 score")
        plt.title("Top 15 configs by F1")
        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        print(f"Plot saved to {os.path.abspath(PLOT_FILE)}")
        plt.show()

if __name__ == "__main__":
    main()
