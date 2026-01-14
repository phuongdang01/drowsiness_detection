import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Dùng để lưu model

# 1. Tải và chuẩn bị dữ liệu
# Định nghĩa tên cho các cột
column_names = ['ear', 'mar', 'pitch', 'yaw', 'roll', 'label']

# Đọc file CSV, báo cho pandas là file KHÔNG CÓ header (header=None)
# và gán tên cột bằng tay (names=column_names)
data = pd.read_csv('drowsiness_data.csv', header=None, names=column_names)

# Xóa các dòng có giá trị NaN (nếu có)
data = data.dropna() 

# Xác định X (đặc trưng) và y (nhãn)
features = ['ear', 'mar', 'pitch', 'yaw', 'roll']
target = 'label'

X = data[features]
y = data[target]

# 2. Chia dữ liệu thành tập Train và Test
# 80% để train, 20% để kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa (Scale) dữ liệu
# Rất quan trọng! Vì EAR (0.3) và Pitch (25.0) có thang đo quá khác nhau
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện Model
print("Bắt đầu huấn luyện Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("Huấn luyện hoàn tất!")

# 5. Đánh giá Model
y_pred = model.predict(X_test_scaled)

print("\n--- Kết quả đánh giá trên tập Test ---")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Tỉnh táo (0)', 'Buồn ngủ (1)']))

# 6. LƯU MODEL VÀ SCALER (Quan trọng nhất)
joblib.dump(model, 'drowsiness_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nĐã lưu model và scaler thành công!")