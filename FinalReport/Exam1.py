import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

import zipfile
import os

zip_file_path = 'data.zip'
extracted_folder_path = 'extracted_data'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

#print(os.listdir(extracted_folder_path))
# ./data/hymenoptera_data/

data_dir = 'extracted_data\data\hymenoptera_data'

image_paths = []
labels = []

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image_paths.append(image_path)
        labels.append(class_name)
#print(labels)

train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

#print(f'train data count: {len(train_image_paths)}')
#print(f'test data count: {len(val_image_paths)}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(root=data_dir, transform=transform)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset.targets)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

import torch.nn as nn
import torch.optim as optim

class CNNNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 111 * 111, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 111 * 111)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#model = CNNNet()
model = CNNNet(input_channels=3, num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'epoch {epoch+1}/{num_epochs}, accuracy: {accuracy:.4f}%')