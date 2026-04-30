import torch
import numpy as np
import cv2
import sys
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.autoencoder import ConvAutoencoder
from torch.utils.data import DataLoader
from dataset import MVTecCapsuleTrainDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_capsule_V3.pth", map_location=device))
model.eval()

def compute_threshold(train_path):
    train_dataset = MVTecCapsuleTrainDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    scores = []

    with torch.no_grad():
        for imgs in train_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            pixel_error = torch.mean((outputs - imgs) ** 2, dim=1)
            pixel_error = pixel_error.view(-1)

            k = int(0.01 * len(pixel_error))
            topk_vals, _ = torch.topk(pixel_error, k)

            loss = torch.mean(topk_vals).item()
            scores.append(loss)

    scores = np.array(scores)

    return np.percentile(scores, 99)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)


pixel_error = torch.mean((output - input_tensor) ** 2, dim=1)
error_map = pixel_error.squeeze().cpu().numpy()

flat_error = pixel_error.view(-1)
k = int(0.01 * len(flat_error))
topk_vals, _ = torch.topk(flat_error, k)
loss = torch.mean(topk_vals).item()


threshold = 0.0066
print("Manual Threshold:", threshold)
print("Anomaly Score:", loss)

if loss > threshold:
    prediction = "DEFECT"
else:
    prediction = "GOOD"

print("Prediction:", prediction)


heatmap = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

orig = cv2.imread(image_path)
orig = cv2.resize(orig, (256, 256))

overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Prediction: {prediction}")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()