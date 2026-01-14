import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

# Define a simple CNN
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128) # Assuming input 64x64 -> 8x8 after 3 pools
        self.fc2 = nn.Linear(128, 2) # 2 classes: Closed, Open
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

class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, mapping):
        self.subset = subset
        self.mapping = mapping
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.mapping[y]
    def __len__(self):
        return len(self.subset)

def train():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 5
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load Data
    data_dir = 'dataset_eyes&yawn/train'
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Get indices of Open and Closed classes
    classes = full_dataset.class_to_idx
    print("Classes found:", classes)
    
    target_classes = ['Closed', 'Open']
    # Check if classes exist
    for c in target_classes:
        if c not in classes:
            print(f"Class {c} not found in dataset.")
            return

    target_indices = [classes[c] for c in target_classes]
    
    # Filter dataset
    indices = [i for i, label in enumerate(full_dataset.targets) if label in target_indices]
    subset_dataset = Subset(full_dataset, indices)
    
    # Mapping: Original Label -> New Label (0 or 1)
    # Closed -> 0, Open -> 1
    mapping = {classes['Closed']: 0, classes['Open']: 1}
    
    train_dataset = RemappedDataset(subset_dataset, mapping)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device} with {len(train_dataset)} images.")

    # Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    print("Finished Training")
    torch.save(model.state_dict(), 'eye_model.pth')
    print("Model saved to eye_model.pth")

if __name__ == '__main__':
    train()
