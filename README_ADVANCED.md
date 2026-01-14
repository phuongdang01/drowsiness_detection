# Advanced Drowsiness Detection System - Hướng dẫn sử dụng

## 📋 Tổng quan

Hệ thống phát hiện buồn ngủ nâng cao với các tính năng:

1. **Segmentation-based Eye Detection**: Sử dụng kiến trúc segmentation để phát hiện mắt chính xác hơn
2. **Multi-task Learning**: Học đồng thời nhận diện mắt và ngáp
3. **Yawn Frequency Analysis**: Phân tích tần suất ngáp trong khoảng thời gian ngắn
4. **Head Pose Estimation**: Ước tính tư thế đầu với hệ tọa độ 3D
5. **Head Nodding Detection**: Phát hiện gục đầu buồn ngủ

## 🗂️ Datasets được sử dụng

- **CEW**: Closed Eyes in the Wild
- **dataset_eyes&yawn**: Dataset chứa cả mắt và ngáp
- **mrleyedataset**: MRL Eye Dataset
- **dataset_nthuddd2**: NTHU Driver Drowsiness Dataset
- **Video Database**: Các video thực tế (25+ videos)

## 🚀 Quy trình sử dụng

### Bước 1: Train Model

```bash
python train_advanced_model.py
```

Script này sẽ:
- Load tất cả datasets từ CEW, mrleyedataset, dataset_eyes&yawn
- Kết hợp thành một dataset lớn
- Train model với kiến trúc segmentation-based
- Multi-task learning: eye state + yawn detection
- Save model tốt nhất: `advanced_drowsiness_model.pth`

**Thời gian train**: ~20 epochs, tùy thuộc vào GPU/CPU

### Bước 2: Extract frames từ Video Database (Optional)

```bash
python extract_video_frames.py
```

Script này giúp:
- Trích xuất frames từ các video trong Video Database
- Lưu frames để có thể label và train thêm

### Bước 3: Chạy Detection System

```bash
python advanced_drowsiness_detection.py
```

## 🎯 Các tính năng chính

### 1. Segmentation-based Eye Detection
- Kiến trúc encoder-decoder giống U-Net
- Feature extraction sâu với 3 tầng encoding
- Bottleneck layer với 512 filters
- Global pooling cho classification

### 2. Eye State Classification
```
Closed Eyes (0): Mắt nhắm
Open Eyes (1): Mắt mở
```
- Sử dụng CNN với batch normalization
- Dropout 0.5 để tránh overfitting
- Confidence threshold: 0.65

### 3. Yawn Frequency Detection
```python
YAWN_WINDOW = 30  # 30 seconds
MAX_YAWNS_IN_WINDOW = 3  # Cảnh báo nếu ngáp >= 3 lần trong 30s
```
- Track timestamps của các lần ngáp
- Tính tần suất trong cửa sổ thời gian
- Cảnh báo nếu vượt ngưỡng

### 4. Head Pose Estimation
- Ước tính góc Pitch, Yaw, Roll của đầu
- Vẽ hệ tọa độ 3D trên đầu:
  - **Trục X** (Đỏ): Hướng ngang
  - **Trục Y** (Xanh lá): Hướng dọc
  - **Trục Z** (Xanh dương): Hướng ra trước

### 5. Head Nodding Detection
```python
HEAD_PITCH_THRESH = 15  # degrees
HEAD_NOD_FRAMES = 20  # frames
```
- Phát hiện khi đầu cúi xuống (pitch > 15°)
- Duy trì >= 20 frames → Cảnh báo gục đầu

## 📊 Cấu trúc Model

```
AdvancedDrowsinessModel
├── Encoder Block 1: Conv(3→64) → BN → ReLU → Conv(64→64) → MaxPool
├── Encoder Block 2: Conv(64→128) → BN → ReLU → Conv(128→128) → MaxPool
├── Encoder Block 3: Conv(128→256) → BN → ReLU → Conv(256→256) → MaxPool
├── Bottleneck: Conv(256→512) → BN → ReLU → Conv(512→512)
├── Global Average Pooling
├── Eye Classifier: FC(512→256) → ReLU → Dropout → FC(256→2)
└── Yawn Classifier: FC(512→256) → ReLU → Dropout → FC(256→2)
```

## ⚙️ Các tham số có thể điều chỉnh

Trong `advanced_drowsiness_detection.py`:

```python
# Thresholds
EAR_THRESH = 0.22          # Ngưỡng EAR cho mắt nhắm
EAR_FRAMES = 15            # Số frames liên tục mắt nhắm
MAR_THRESH = 0.6           # Ngưỡng MAR cho ngáp
YAWN_FRAMES = 20           # Số frames tối thiểu cho 1 cú ngáp
YAWN_WINDOW = 30           # Cửa sổ thời gian (giây)
MAX_YAWNS_IN_WINDOW = 3    # Số lần ngáp tối đa trong cửa sổ
HEAD_PITCH_THRESH = 15     # Góc pitch cảnh báo gục đầu
HEAD_NOD_FRAMES = 20       # Frames liên tục gục đầu
CONFIDENCE_THRESH = 0.65   # Confidence tối thiểu cho prediction

# Video source
USE_VIDEO_FILE = True      # True: dùng video, False: dùng webcam
VIDEO_PATH = r"Video Database\Sub 01.avi"  # Đường dẫn video
```

## 🎨 Giao diện hiển thị

1. **EAR (Eye Aspect Ratio)**: Tỷ lệ mắt
2. **MAR (Mouth Aspect Ratio)**: Tỷ lệ miệng
3. **Yawn Freq**: Tần suất ngáp (x/3)
4. **L-Eye / R-Eye**: Trạng thái mắt trái/phải với confidence
5. **Pitch/Yaw/Roll**: Góc quay đầu
6. **Hệ tọa độ 3D**: Vẽ trên đầu người
7. **Bounding boxes**: 
   - Xanh: Mắt mở / Không ngáp
   - Đỏ: Mắt nhắm / Đang ngáp

## 🚨 Cảnh báo buồn ngủ

Hệ thống cảnh báo khi phát hiện:
1. **EYES CLOSED**: Mắt nhắm liên tục
2. **FREQUENT YAWNING**: Ngáp quá nhiều trong thời gian ngắn
3. **HEAD NODDING**: Gục đầu

## 📈 Cải thiện trong tương lai

1. **Attention mechanism**: Thêm attention vào model
2. **LSTM/GRU**: Sử dụng temporal features
3. **Data augmentation**: Tăng cường dữ liệu từ video
4. **Real-time optimization**: Tối ưu tốc độ inference
5. **Mobile deployment**: Deploy lên mobile devices

## 🔧 Troubleshooting

### Model không tìm thấy
```
⚠️ Advanced model not found, using basic model
```
→ Chạy `train_advanced_model.py` để train model

### Alert sound không phát
```
Alert sound not found!
```
→ Thêm file `alert.wav` vào thư mục chính

### Video không mở được
```
❌ Error: Cannot open video file
```
→ Kiểm tra đường dẫn video trong `VIDEO_PATH`

### GPU không được sử dụng
```
🖥️ Using device: cpu
```
→ Cài đặt CUDA và PyTorch với GPU support

## 📝 Requirements

```bash
pip install torch torchvision opencv-python mediapipe numpy scipy pillow playsound scikit-learn tqdm
```

## 🤝 Contributing

Để cải thiện hệ thống:
1. Thu thập thêm dữ liệu từ Video Database
2. Label chính xác các trạng thái drowsy/alert
3. Fine-tune các thresholds
4. Thử nghiệm các kiến trúc model khác

---

**Tác giả**: Advanced Drowsiness Detection System
**Version**: 2.0
**Ngày**: 2026-01-02
