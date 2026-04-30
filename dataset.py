import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# Common image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



class MVTecCapsuleTrainDataset(Dataset):
    def __init__(self, root_dir):

        self.samples = []

        train_path = os.path.join(root_dir, "train", "good")

        for file in os.listdir(train_path):
            if file.endswith(".png"):
                self.samples.append(os.path.join(train_path, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        return image


class MVTecCapsuleTestDataset(Dataset):
    def __init__(self, root_dir):

        self.samples = []

        test_path = os.path.join(root_dir, "test")

        for category in os.listdir(test_path):

            category_path = os.path.join(test_path, category)

            if not os.path.isdir(category_path):
                continue

            for file in os.listdir(category_path):

                if not file.endswith(".png"):
                    continue

                img_path = os.path.join(category_path, file)

                label = 0 if category == "good" else 1

                mask_path = None

                if label == 1:
                    mask_path = os.path.join(
                        root_dir,
                        "ground_truth",
                        category,
                        file.replace(".png", "_mask.png")
                    )

                self.samples.append((img_path, label, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = transform(mask)
        else:
            mask = torch.zeros((1, 256, 256))

        return image, label, mask