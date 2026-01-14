"""
Advanced Drowsiness Detection Model Training
- Combines multiple datasets: CEW, dataset_eyes&yawn, mrleyedataset, dataset_nthuddd2
- Uses segmentation-based eye detection
- Multi-task learning: eye state + yawn detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

# ====================== DATASET CLASS ======================
class MultiDataset(Dataset):
    """Unified dataset loader for all eye/yawn datasets"""
    def __init__(self, data_list, transform=None):
        """
        data_list: list of (image_path, label) tuples
        label format: {'eye': 0/1, 'yawn': 0/1} where 0=closed/yawn, 1=open/no_yawn
        """
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Return image and labels
            return image, label['eye'], label['yawn']
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image
            dummy = torch.zeros(3, 64, 64)
            return dummy, 1, 1  # default: open, no_yawn

# ====================== DATA LOADER ======================
def load_all_datasets():
    """Load and combine all available datasets"""
    data_list = []
    
    print("📂 Loading datasets...")
    
    # 1. CEW Dataset (Closed Eyes in the Wild)
    print("  - CEW dataset...")
    cew_closed = glob.glob(r"CEW/closed/**/*.jpg", recursive=True) + \
                 glob.glob(r"CEW/closed/**/*.png", recursive=True)
    cew_open = glob.glob(r"CEW/open/**/*.jpg", recursive=True) + \
               glob.glob(r"CEW/open/**/*.png", recursive=True)
    
    for img in cew_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))  # closed, no_yawn
    for img in cew_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))  # open, no_yawn
    print(f"    CEW: {len(cew_closed)} closed, {len(cew_open)} open")
    
    # 2. MRL Eye Dataset
    print("  - MRL Eye dataset...")
    mrl_closed = glob.glob(r"mrleyedataset/Close-Eyes/**/*.jpg", recursive=True) + \
                 glob.glob(r"mrleyedataset/Close-Eyes/**/*.png", recursive=True)
    mrl_open = glob.glob(r"mrleyedataset/Open-Eyes/**/*.jpg", recursive=True) + \
               glob.glob(r"mrleyedataset/Open-Eyes/**/*.png", recursive=True)
    
    for img in mrl_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))
    for img in mrl_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    print(f"    MRL: {len(mrl_closed)} closed, {len(mrl_open)} open")
    
    # 3. Dataset Eyes & Yawn (Training set)
    print("  - Eyes & Yawn dataset...")
    ey_train_closed = glob.glob(r"dataset_eyes&yawn/train/Closed/**/*.jpg", recursive=True) + \
                      glob.glob(r"dataset_eyes&yawn/train/Closed/**/*.png", recursive=True)
    ey_train_open = glob.glob(r"dataset_eyes&yawn/train/Open/**/*.jpg", recursive=True) + \
                    glob.glob(r"dataset_eyes&yawn/train/Open/**/*.png", recursive=True)
    ey_train_yawn = glob.glob(r"dataset_eyes&yawn/train/yawn/**/*.jpg", recursive=True) + \
                    glob.glob(r"dataset_eyes&yawn/train/yawn/**/*.png", recursive=True)
    ey_train_no_yawn = glob.glob(r"dataset_eyes&yawn/train/no_yawn/**/*.jpg", recursive=True) + \
                       glob.glob(r"dataset_eyes&yawn/train/no_yawn/**/*.png", recursive=True)
    
    for img in ey_train_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))
    for img in ey_train_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    for img in ey_train_yawn:
        data_list.append((img, {'eye': 1, 'yawn': 0}))  # assume open when yawning
    for img in ey_train_no_yawn:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    
    print(f"    Eyes&Yawn: {len(ey_train_closed)} closed, {len(ey_train_open)} open")
    print(f"               {len(ey_train_yawn)} yawn, {len(ey_train_no_yawn)} no_yawn")
    
    print(f"\n✅ Total samples: {len(data_list)}")
    return data_list

# ====================== ADVANCED MODEL ======================
class AdvancedDrowsinessModel(nn.Module):
    """
    Multi-task model with:
    - Segmentation-inspired feature extraction
    - Eye state classification
    - Yawn detection
    """
    def __init__(self):
        super(AdvancedDrowsinessModel, self).__init__()
        
        # Encoder (Feature Extraction with Segmentation-like structure)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classification heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Eye state classifier
        self.eye_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Closed/Open
        )
        
        # Yawn detector
        self.yawn_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Yawn/No_Yawn
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x = self.pool3(x3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling
        features = self.global_pool(x)
        
        # Classification
        eye_out = self.eye_classifier(features)
        yawn_out = self.yawn_classifier(features)
        
        return eye_out, yawn_out

# ====================== TRAINING FUNCTION ======================
def train_model():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load data
    data_list = load_all_datasets()
    
    # Split data
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    train_dataset = MultiDataset(train_data, transform=transform)
    val_dataset = MultiDataset(val_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Model
    model = AdvancedDrowsinessModel().to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    
    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_eye_correct = 0
        train_yawn_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, eye_labels, yawn_labels in pbar:
            images = images.to(device)
            eye_labels = eye_labels.to(device)
            yawn_labels = yawn_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            eye_out, yawn_out = model(images)
            
            # Loss
            loss_eye = criterion(eye_out, eye_labels)
            loss_yawn = criterion(yawn_out, yawn_labels)
            loss = loss_eye + loss_yawn
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            _, eye_pred = torch.max(eye_out, 1)
            _, yawn_pred = torch.max(yawn_out, 1)
            train_eye_correct += (eye_pred == eye_labels).sum().item()
            train_yawn_correct += (yawn_pred == yawn_labels).sum().item()
            train_total += images.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_eye_acc = 100 * train_eye_correct / train_total
        train_yawn_acc = 100 * train_yawn_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_eye_correct = 0
        val_yawn_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, eye_labels, yawn_labels in val_loader:
                images = images.to(device)
                eye_labels = eye_labels.to(device)
                yawn_labels = yawn_labels.to(device)
                
                eye_out, yawn_out = model(images)
                
                loss_eye = criterion(eye_out, eye_labels)
                loss_yawn = criterion(yawn_out, yawn_labels)
                loss = loss_eye + loss_yawn
                
                val_loss += loss.item()
                _, eye_pred = torch.max(eye_out, 1)
                _, yawn_pred = torch.max(yawn_out, 1)
                val_eye_correct += (eye_pred == eye_labels).sum().item()
                val_yawn_correct += (yawn_pred == yawn_labels).sum().item()
                val_total += images.size(0)
        
        val_loss /= len(val_loader)
        val_eye_acc = 100 * val_eye_correct / val_total
        val_yawn_acc = 100 * val_yawn_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Eye Acc: {train_eye_acc:.2f}% | Yawn Acc: {train_yawn_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Eye Acc: {val_eye_acc:.2f}% | Yawn Acc: {val_yawn_acc:.2f}%")
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'advanced_drowsiness_model.pth')
            print(f"  ✅ Best model saved!")
        
        print()
    
    print("🎉 Training completed!")
    print(f"📦 Model saved as: advanced_drowsiness_model.pth")

if __name__ == "__main__":
    train_model()
