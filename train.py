import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MVTecCapsuleTrainDataset
from models.autoencoder import ConvAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset_path = "dataset/capsule"

train_dataset = MVTecCapsuleTrainDataset(dataset_path)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30

for epoch in range(epochs):
    total_loss = 0

    for imgs in train_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "autoencoder_capsule_V3.pth")
print("Model Saved Successfully!")