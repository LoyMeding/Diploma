import torch
import numpy as np
import torch.utils.data

from tqdm import tqdm
from typing import List


'''
 Объявляем класс для создания базовой модели, которые могут быть использованы в последствии для стэкинга и бэгинга
'''


class PytorchModel:
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
    def fit(self, train_loader):
        if self.cuda:
            self.net.cuda()
        if self.pretrained_path:
            self.net.load_state_dict(torch.load(self.pretrained_path))
        self.optim = self.optim_type(self.net.parameters(), **self.optim_params)
        self.net.train()

        print(f"---------------Start Training {self.model_name}----------------")
        for epoch in tqdm(range(self.epochs)):
            train_samples_count = 0
            true_train_samples_count = 0
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
            torch.save(self.net.state_dict(), "/content/drive/MyDrive/Models/" + self.model_name + str(epoch) + ".pt")
            print(f"Epoch {epoch}, accuracy: {train_accuracy}, loss: {loss}")
        print("---------------End Training--------------------")

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


def train(
        model,
        optim,
        optim_params,
        loss_fn,
        image_shape: List[int],
        classes_: List[str],
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        pretrained_path: str = None,
        model_name: str = None,
        batch_size: int = 32,
        cuda: bool = False,
):
    base_model = PytorchModel(net=model, optim_type=optim, epochs=epochs, optim_params=optim_params, loss_fn=loss_fn,
                              pretrained_path=pretrained_path, model_name=model_name,
                              batch_size=batch_size, cuda=cuda, image_shape=image_shape, classes_=classes_)
    base_model.fit(train_loader)


def predict(
        model,
        optim,
        optim_params,
        loss_fn,
        image_shape: List[int],
        classes_: List[str],
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        pretrained_path: str = None,
        model_name: str = None,
        batch_size: int = 32,
        cuda: bool = False,
):
    base_model = PytorchModel(net=model, optim_type=optim, epochs=epochs, optim_params=optim_params, loss_fn=loss_fn,
                              pretrained_path=pretrained_path, model_name=model_name,
                              batch_size=batch_size, cuda=cuda, image_shape=image_shape, classes_=classes_)
    base_model.predict(test_loader)
