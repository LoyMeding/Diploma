import torch
import os
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image
from typing import Tuple
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset, Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.images = []
        self.labels = []
        self.transform = transform

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            images = os.listdir(class_dir)
            for image_name in images:
                self.images.append(os.path.join(class_dir, image_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


'''
    Создаём список аугментаций, импортируем данные и разбиваем на обучающую, валидационную и тестовую выборки,
    создаём загрузчики данных для выборок.
'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((250, 250), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.7, 1.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def dataloaders(
        batch_size: int,
        shuffle: bool,
        train_dir: str,
        test_dir: str,
        valid_dir: str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = ImageFolderDataset(train_dir, transform=transform)
    test_dataset = ImageFolderDataset(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
