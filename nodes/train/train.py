import torch
import numpy as np
import torch.utils.data
import sklearn

from tqdm import tqdm
from typing import List

from torch import nn
from torch.optim import Adam

"""
Объявляем класс для создания базовой модели, описываем для класса методы инициализации, обучения и предсказания

Параметры для инициализации:

- net: модель нейронной сети, которую нужно обучить.
- optim_type: тип оптимизатора, используемого для обновления весов модели.
- optim_params: дополнительные параметры, специфичные для выбранного оптимизатора.
- loss_fn: функция потерь, используемая для вычисления разницы между предсказанными и истинными значениями.
- epochs: количество проходов через всю обучающую выборку.
- batch_size: количество образцов в каждой мини-партии во время обучения.
- image_shape: форма входных изображений, подаваемых на модель.
- cuda: флаг, позволяющий использовать ускорение на GPU или использовать только CPU.
- classes_: список меток классов или категорий для задач классификации.
- pretrained_path: путь к заранее обученной модели, если таковая имеется.
- model_name: имя или идентификатор модели.
"""


class PytorchModel(sklearn.base.BaseEstimator):
    def __init__(self, net, optim_type, optim_params, loss_fn, classes_,
                 epochs, image_shape, pretrained_path=None, model_name=None, batch_size=32,
                 cuda=False):
        self.net = net
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.cuda = cuda
        self.classes_ = classes_
        self.pretrained_path = pretrained_path
        self.model_name = model_name

    # Описываем функцию для обучения модели
    def fit(self, train_loader, valid_loader=None):
        if self.cuda:
            self.net.cuda()
        if self.pretrained_path != "None":
            self.net.load_state_dict(torch.load(self.pretrained_path))
        self.optim = self.optim_type(self.net.parameters(), **self.optim_params)
        self.net.train()

        train_accuracy_list = []
        valid_accuracy_list = []
        loss_list = []

        print(f"---------------Start Training {self.model_name}----------------")
        for epoch in tqdm(range(self.epochs)):
            train_samples_count = 0
            true_train_samples_count = 0
            valid_samples_count = 0
            true_valid_samples_count = 0
            valid_accuracy = None
            for x, y in train_loader:
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                y_pred = self.net(x)
                loss = self.loss_fn(y_pred, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                y_pred = y_pred.argmax(dim=1, keepdim=False)
                true_classified = (y_pred == y).sum().item()
                true_train_samples_count += true_classified
                train_samples_count += len(x)
            train_accuracy = true_train_samples_count / train_samples_count

            if valid_loader:
                for x, y in valid_loader:
                    if self.cuda:
                        x = x.cuda()
                        y = y.cuda()
                    y_pred = self.net(x)
                    y_pred = y_pred.argmax(dim=1, keepdim=False)
                    true_classified = (y_pred == y).sum().item()
                    true_valid_samples_count += true_classified
                    valid_samples_count += len(x)
                valid_accuracy = true_valid_samples_count / valid_samples_count

            train_accuracy_list.append(train_accuracy)
            valid_accuracy_list.append(valid_accuracy)
            loss_list.append(loss.detach().cpu().numpy())
            print(f"Epoch {epoch},Train accuracy: {train_accuracy}, Valid accuracy: {valid_accuracy}, Loss: {loss}")
        print("---------------End Training--------------------")
        return train_accuracy_list, valid_accuracy_list, loss_list

    # Описываем функцию для предсказания вероятностей классов
    def predict_proba(self, test_loader):
        self.net.eval()
        predictions = []
        for x_batch, y_batch in tqdm(test_loader):
            if self.cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            y_pred = self.net(x_batch)
            predictions.append(y_pred.detach().cpu().numpy())
        predictions = np.concatenate(predictions)
        return predictions

    # Описываем функцию для предсказания класса
    def predict(self, test_loader):
        predictions = self.predict_proba(test_loader)
        predictions = predictions.argmax(axis=1)
        return predictions

"""
Функция для обучения модели. Создаёт экземпляр базового класса и, используя его метод fit, производит обучение модели

Параметры:
- model: модель для обучения.
- image_shape: размер изображений в формате [высота, ширина, количество каналов].
- classes_: список классов.
- train_loader: загрузчик данных для обучения.
- epochs: количество эпох обучения (по умолчанию 10).
- pretrained_path: путь к предобученной модели (по умолчанию None).
- model_name: название модели (по умолчанию None).
- batch_size: размер пакета данных (по умолчанию 32).
- cuda: использование GPU для обучения (по умолчанию False).
"""


def train(
        model,
        image_shape: List[int],
        classes_: List[str],
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        pretrained_path: str = None,
        model_name: str = None,
        batch_size: int = 32,
        cuda: bool = False,
):
    base_model = PytorchModel(net=model, optim_type=Adam, epochs=epochs, optim_params={"lr": 1e-3},
                              loss_fn=nn.CrossEntropyLoss(),
                              pretrained_path=pretrained_path, model_name=model_name,
                              batch_size=batch_size, cuda=cuda, image_shape=image_shape, classes_=classes_)
    base_model.fit(train_loader)

    """
    Функция для получения предсказаний модели. Создаёт экземпляр базового класса и, используя его метод predict, выдаёт предсказания модели

    Параметры:
    - model: модель для обучения.
    - image_shape: размер изображений в формате [высота, ширина, количество каналов].
    - classes_: список классов.
    - test_loader: загрузчик данных для предсказания.
    - epochs: количество эпох обучения (по умолчанию 10).
    - pretrained_path: путь к предобученной модели (по умолчанию None).
    - model_name: название модели (по умолчанию None).
    - batch_size: размер пакета данных (по умолчанию 32).
    - cuda: использование GPU для обучения (по умолчанию False).
    """


def predict(
        model,
        image_shape: List[int],
        classes_: List[str],
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        pretrained_path: str = None,
        model_name: str = None,
        batch_size: int = 32,
        cuda: bool = False,
):
    base_model = PytorchModel(net=model, optim_type=Adam, epochs=epochs, optim_params={"lr": 1e-3},
                              loss_fn=nn.CrossEntropyLoss(),
                              pretrained_path=pretrained_path, model_name=model_name,
                              batch_size=batch_size, cuda=cuda, image_shape=image_shape, classes_=classes_)
    base_model.predict(test_loader)
