import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from dataset import MVTecCapsuleTestDataset
from models.autoencoder import ConvAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "C:/Users/vrajs/Desktop/Final_year/capsule_anomaly_detection/dataset/capsule"

test_dataset = MVTecCapsuleTestDataset(dataset_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_capsule.pth"))
model.eval()

all_scores = []
all_labels = []
all_pixel_preds = []
all_pixel_labels = []

with torch.no_grad():
    for imgs, labels, masks in test_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)


        loss = torch.mean((outputs - imgs) ** 2, dim=[1,2,3])
        score = loss.item()

        all_scores.append(score)
        all_labels.append(labels.item())

        pixel_error = torch.mean((outputs - imgs) ** 2, dim=1)
        error_map = pixel_error.squeeze().cpu().numpy()

        gt_mask = masks[0].numpy()

        gt_mask = (gt_mask > 0.5).astype(int)

        gt_mask = gt_mask.flatten()
        pred_mask = error_map.flatten()

        all_pixel_labels.extend(gt_mask)
        all_pixel_preds.extend(pred_mask)


roc_auc = roc_auc_score(all_labels, all_scores)
print("Image-level ROC-AUC:", roc_auc)


threshold = np.percentile(all_scores, 95)
print("Threshold:", threshold)

predictions = [1 if s > threshold else 0 for s in all_scores]

print("Confusion Matrix:")
print(confusion_matrix(all_labels, predictions))

print("Classification Report:")
print(classification_report(all_labels, predictions))


if len(all_pixel_labels) > 0:
    pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_preds)
    print("Pixel-level ROC-AUC:", pixel_auc)