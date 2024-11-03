import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Step 1: Define the data transforms (e.g., normalization, resizing, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256 (adjust as needed)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Step 2: Load the dataset using ImageFolder
# Make sure your dataset is in the specified folder structure
data_dir = 'C:/Users/Tiggs_/PycharmProjects/yolov8ModelTest/Training_The_Model/dataset'
train_dataset = ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Get the number of classes (folders) in your dataset
num_classes = len(train_dataset.classes)
print(f'Classes: {train_dataset.classes}')
print(f'Number of classes: {num_classes}')


# Step 3: Define a simple CNN model (adjust as needed)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Calculate the size after conv layers and pooling to define fc1 input size
        self.fc1 = nn.Linear(32 * 62 * 62, 128)  # Update based on the output size after conv and pooling
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Apply pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Apply pooling
        print("Shape after conv layers:", x.shape)  # Debugging line
        x = x.view(-1, 32 * 62 * 62)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Initialize the model with the correct number of classes
model = SimpleCNN(num_classes=num_classes)

# Step 4: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
num_epochs = 5 #number of complete passes through the entire training dataset
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Step 6: Save the trained model as a .pt file
torch.save(model.state_dict(), 'C:Users/Tiggs_/PycharmProjects/yolov8ModelTest/Training_The_Model/Models/modelTest.pt')

print("Model saved as 'model.pth'")
