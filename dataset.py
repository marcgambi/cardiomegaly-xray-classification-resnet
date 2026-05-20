import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """
    Custom dataset for chest X-ray binary classification.
    """

    def __init__(self, image_paths, targets, augment=False):
        self.image_paths = image_paths
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

        image = self._preprocess_image(image)

        if self.augment:
            image = self._apply_augmentation(image)

        image = torch.tensor(image).float()
        image = image.unsqueeze(0)

        target = torch.tensor(self.targets[index]).float()

        return image, target

    def _preprocess_image(self, image):
        """
        Apply preprocessing steps before model training.
        """

        image = self._center_window(image, window_size=700)
        image = cv2.resize(image, (512, 512))
        image = cv2.equalizeHist(image)

        return image / 255.0

    def _center_window(self, image, window_size):
        """
        Crop a centered window from the image.
        """

        height, width = image.shape[:2]

        center_x = width // 2
        center_y = height // 2
        half_size = window_size // 2

        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size, width)

        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size, height)

        return image[start_y:end_y, start_x:end_x]

    def _apply_augmentation(self, image):
        """
        Apply simple data augmentation to improve model robustness.
        """

        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(
                image,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

        if random.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image
