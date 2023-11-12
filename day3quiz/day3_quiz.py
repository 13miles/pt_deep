import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=48)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class IrisClassification(nn.Module):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisClassification()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
accuracies = []

for epoch in range(100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_func(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predict = torch.max(output, 1)[1]
        total += y_batch.size(0)
        correct += (predict == y_batch).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)

    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"epoch [{epoch+1}/100], loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        output = model(x_batch)
        predict = torch.max(output, 1)[1]
        total += y_batch.size(0)
        correct += (predict == y_batch).sum().item()

accuracy = correct / total
print(f"test result - accuracy: {accuracy:.4f}")

plt.figure(figsize=(8, 8))
plt.plot(losses, color='r', label='loss')
plt.plot(accuracies, color='b', label='accuracy')
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()