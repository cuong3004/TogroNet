import torch.nn as nn
import kornia.augmentation as K


class SSLKorniaAugment(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.aug = nn.Sequential(
            K.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.08, 1.0),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(
                kernel_size=(23, 23),
                sigma=(0.1, 2.0),
                p=0.5,
            ),
        )

    def forward(self, x):
        return self.aug(x)
