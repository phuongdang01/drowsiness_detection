# Drowsiness Detection System with Deep Learning

## Mô tả hệ thống

Hệ thống phát hiện buồn ngủ cho tài xế sử dụng deep learning (PyTorch) và computer vision để:
- **Nhận diện mắt** bằng mô hình CNN đã được train
- **Tính toán EAR (Eye Aspect Ratio)** và **MAR (Mouth Aspect Ratio)**
- **Đưa ra cảnh báo** khi phát hiện tài xế đang buồn ngủ

## Các file chính

### 1. `train_eye_model_torch.py`
File huấn luyện mô hình CNN để phân loại mắt mở/đóng.

**Kiến trúc mô hình:**
- 3 lớp Convolutional (32, 64, 128 filters)
- MaxPooling sau mỗi lớp conv
- 2 lớp Fully Connected (128, 2)
- Dropout để tránh overfitting

**Dataset:** `dataset_eyes&yawn/train/`
- Closed: Mắt đóng (~617 ảnh)
- Open: Mắt mở (~617 ảnh)

**Kết quả training:**
```
Epoch 1, Loss: 0.4289
Epoch 2, Loss: 0.1840
Epoch 3, Loss: 0.1172
Epoch 4, Loss: 0.0721
Epoch 5, Loss: 0.0708
```

**Output:** `eye_model.pth` - Model đã được train

### 2. `drowsiness_detection_with_model.py`
File chính để chạy hệ thống phát hiện buồn ngủ real-time.

**Các tính năng:**
1. **Phát hiện mắt bằng Deep Learning:**
   - Trích xuất vùng mắt từ landmarks của MediaPipe
   - Dự đoán trạng thái mắt (mở/đóng) bằng CNN model
   - Tính độ tin cậy (confidence) của dự đoán

2. **Tính toán EAR (Eye Aspect Ratio):**
   - Công thức: `EAR = (A + B) / (2.0 * C)`
   - Ngưỡng: `EAR_THRESH = 0.22`
   - Số frame cảnh báo: `EAR_FRAMES = 15` (~0.5 giây)

3. **Tính toán MAR (Mouth Aspect Ratio):**
   - Công thức: `MAR = A / B`
   - Phát hiện ngáp (yawning)
   - Ngưỡng: `MAR_THRESH = 0.6`
   - Số frame cảnh báo: `MAR_FRAMES = 15`

4. **Cảnh báo:**
   - Hiển thị cảnh báo màu đỏ trên màn hình
   - Phát âm thanh cảnh báo (`alert.wav`) khi phát hiện buồn ngủ
   - Tự động dừng cảnh báo khi tài xế tỉnh táo

**Các ngưỡng:**
```python
EAR_THRESH = 0.22           # Ngưỡng EAR
EAR_FRAMES = 15             # Số frame liên tiếp để kích hoạt cảnh báo
MAR_THRESH = 0.6            # Ngưỡng MAR (ngáp)
MAR_FRAMES = 15             # Số frame ngáp
CONFIDENCE_THRESH = 0.7     # Ngưỡng độ tin cậy của model
```

## Cách chạy

### Bước 1: Huấn luyện model (nếu chưa có)
```bash
python train_eye_model_torch.py
```

### Bước 2: Chạy hệ thống phát hiện
```bash
python drowsiness_detection_with_model.py
```

**Lưu ý:**
- Thay `cap = cv2.VideoCapture(0)` thành `cap = cv2.VideoCapture("video.mp4")` để test với video
- Nhấn `Q` để thoát chương trình

## Hiển thị trên màn hình

Hệ thống hiển thị:
- **EAR**: Giá trị Eye Aspect Ratio
- **MAR**: Giá trị Mouth Aspect Ratio  
- **L-Eye/R-Eye**: Trạng thái mắt trái/phải (OPEN/CLOSED) và độ tin cậy
- **Khung màu**:
  - Xanh: Mắt/miệng bình thường
  - Đỏ: Phát hiện mắt đóng/ngáp
- **Status**:
  - "Status: ALERT" (màu xanh): Tỉnh táo
  - "DROWSY - EYES CLOSED!" (màu đỏ): Phát hiện mắt đóng
  - "DROWSY - YAWNING!" (màu đỏ): Phát hiện ngáp

## Yêu cầu thư viện

```bash
pip install torch torchvision opencv-python mediapipe scipy pillow playsound
```

## Cấu trúc thư mục

```
.
├── train_eye_model_torch.py          # Train model
├── drowsiness_detection_with_model.py # Hệ thống phát hiện chính
├── eye_model.pth                      # Model đã train
├── alert.wav                          # File âm thanh cảnh báo
└── dataset_eyes&yawn/
    └── train/
        ├── Closed/                    # Ảnh mắt đóng
        └── Open/                      # Ảnh mắt mở
```

## Cải tiến so với phương pháp cũ

1. **Segmentation với Deep Learning:** Thay vì chỉ dùng ngưỡng EAR, hệ thống sử dụng CNN để phân loại mắt chính xác hơn
2. **Độ tin cậy:** Model đưa ra confidence score cho mỗi dự đoán
3. **Kết hợp nhiều phương pháp:** Sử dụng cả model AI và các metrics truyền thống (EAR, MAR)
4. **Real-time processing:** Tối ưu hóa để chạy real-time với webcam

## Kết luận

Hệ thống đã được train và test thành công với:
- Model CNN đạt loss < 0.1 sau 5 epochs
- Phát hiện chính xác trạng thái mắt mở/đóng
- Cảnh báo kịp thời khi phát hiện dấu hiệu buồn ngủ
- Hoạt động ổn định với cả webcam và video file
