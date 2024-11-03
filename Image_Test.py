import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os




# Step 1: Define the transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the same size as during training
    transforms.ToTensor(),  # Convert to tensor
])

# Step 2: Load the trained model
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 62 * 62, 128)  # Adjust as needed
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 62 * 62)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the model architecture
num_classes = 7  # Set this to the number of classes in your dataset
model = SimpleCNN(num_classes=num_classes)

# Load the trained weights
model.load_state_dict(torch.load('C:/Users/Tiggs_/PycharmProjects/yolov8ModelTest/Training_The_Model/Models/model_larger_dataset.pt'))
model.eval()  # Set the model to evaluation mode

# Define class names (make sure these match your dataset)
class_names = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Change these to your actual class names

# Step 3: Run inference on an image
def run_inference(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index

    return predicted.item()  # Return the predicted class index

# Example usage
if __name__ == "__main__":
    # Path to the image you want to test
    test_image_path = 'C:/Users/Tiggs_/PycharmProjects/yolov8ModelTest/Training_The_Model/Test_Files/beer_bottle.jpg'  # Change this to your image path

    if os.path.exists(test_image_path):
        predicted_class_index = run_inference(test_image_path)
        predicted_class_name = class_names[predicted_class_index]
        print(f'Predicted class name: {predicted_class_name}')
    else:
        print("The specified image file does not exist.")
