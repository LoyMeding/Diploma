from torchvision.models import resnet50, efficientnet_b0
from torch import nn
from typing import List

'''
Инициализируем предобученную модель ResNet после чего замораживаем первые 45 слоёв
и последний полносвязный слой заменяем на полносвязный слой с количеством выходов
раввным количеству наших классов
'''


def model(
        pretrained: bool,
        classes: List[str]
) -> resnet50:
    resnet = resnet50(pretrained=pretrained)
    # Замораживаем все слои
    for param in resnet.parameters():
        param.requires_grad = False
    num_layers = len(list(resnet.children()))

    # Размораживаем последние 3 слоя
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    # Заменяем последний полносвязный слой
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, len(classes))
    return resnet
