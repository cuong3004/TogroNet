from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T


class ImageFolderSSL(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_paths = []

        image_dir = Path(image_dir)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

        for ext in exts:
            self.image_paths.extend(list(image_dir.rglob(ext)))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        return img, 0


def build_datasets(config):
    train_dataset = ImageFolderSSL(
        image_dir=config["data"]["image_dir_train"],
        image_size=config["data"]["image_size"],
    )

    val_dataset = ImageFolderSSL(
        image_dir=config["data"]["image_dir_valid"],
        image_size=config["data"]["image_size"],
    )

    # val_size = int(len(dataset) * config["data"]["val_ratio"])
    # train_size = len(dataset) - val_size

    # train_dataset, val_dataset = random_split(
    #     dataset,
    #     [train_size, val_size],
    # )

    return train_dataset, val_dataset