import torch
import torch.nn as nn
import os
DEVICE='cuda'
class SimpleInterpolationModel(nn.Module):
    def __init__(self):
        super(SimpleInterpolationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image1, image2):
        x = torch.cat((image1, image2), dim=1)  # Concatenate image1 and image2 along the channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Output interpolated image
    

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

MODEL_PATH=r'Frame_Interpolation\frame_interpolation_model.pth'
INPUT_DIR = r'Frame_Interpolation\output_images'  # Directory with sequential images (e.g., 1.png, 2.png, ...)
OUTPUT_DIR = r'Frame_Interpolation\predicted_frames'  # Directory to save predicted frames
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE=512
# Transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# Load Trained Model
model = SimpleInterpolationModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Function to visualize and save predictions
def predict_and_visualize(image1_path, image2_path, output_path):
    # Load images
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Apply transformations
    image1 = transform(image1).unsqueeze(0).to(DEVICE)  # Add batch dimension
    image2 = transform(image2).unsqueeze(0).to(DEVICE)

    # Predict intermediate frame
    with torch.no_grad():
        predicted_image = model(image1, image2).squeeze(0).cpu()  # Remove batch dimension

    # Convert tensor to image
    predicted_image = transforms.ToPILImage()(predicted_image)

    # Save the predicted image
    predicted_image.save(output_path)

    # # Visualize input and predicted images
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(Image.open(image1_path))
    # axes[0].set_title('Image 1')
    # axes[0].axis('off')
    # axes[1].imshow(predicted_image)
    # axes[1].set_title('Predicted Image')
    # axes[1].axis('off')
    # axes[2].imshow(Image.open(image2_path))
    # axes[2].set_title('Image 2')
    # axes[2].axis('off')
    # plt.show()

# Test on a sequence of images
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

for i in range(len(image_files) - 1):  # Loop through consecutive pairs
    image1_path = os.path.join(INPUT_DIR, image_files[i])
    image2_path = os.path.join(INPUT_DIR, image_files[i+1])
    output_path = os.path.join(OUTPUT_DIR, f'predicted_{i+1}.png')

    predict_and_visualize(image1_path, image2_path, output_path)
    print(f"Predicted frame saved to {output_path}")

print("Prediction completed.")

