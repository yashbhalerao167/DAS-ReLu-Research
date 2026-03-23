# src/dataset.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MURADataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.root_dir = Path(root_dir)

        self.samples = []

        with open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split()

                # Format: "image_path label"
                img_rel_path = parts[0]
                label = int(parts[1])

                self.samples.append((img_rel_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]

        img_path = self.root_dir / img_rel_path

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        return image, label