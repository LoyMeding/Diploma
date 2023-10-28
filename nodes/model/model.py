from torchvision.models import resnet50
from torch import nn
from typing import List

'''
Инициализируем предобученную модель ResNet после чего замораживаем первые 45 слоёв
и последний полносвязный слой заменяем на полносвязный слой с количеством выходов
раввным количеству наших классов
'''


def model(
        classes: List[str]
) -> resnet50:
    resnet = resnet50()
    # Замораживаем все слои
    for param in resnet.parameters():
        param.requires_grad = False

    # Размораживаем последние 3 слоя
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    # Заменяем последний полносвязный слой
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, len(classes))
    return resnet
