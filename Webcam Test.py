import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image

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
model.load_state_dict(torch.load('C:/Users/Tiggs_/Documents/yolov8-silva-main/Training_The_Model/model_larger_dataset.pt'))
model.eval()  # Set the model to evaluation mode

# Step 3: Define the class names (either load from a file or define manually)
class_names = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Replace with your actual class names

# Step 4: Run inference from the webcam
def run_inference(frame):
    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index

    predicted_class_name = class_names[predicted.item()]  # Get the class name
    return predicted_class_name  # Return the predicted class name

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 is the webcam number depending on webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run inference on the captured frame
        predicted_class = run_inference(frame)

        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted class: {predicted_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame with the prediction
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
