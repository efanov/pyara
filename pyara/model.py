"""
Module for where models described
"""
import torch
from torch import nn
import os
from config import CFG


class ResNetBlock(nn.Module):
    """
       Класс, представляющий блок ResNet.

       Параметры:
           in_depth (int): Количество входных каналов.
           depth (int): Количество выходных каналов.
           first (bool, опционально): Является ли блок первым в сети. По умолчанию False.
    """

    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock,
              self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth,
                               kernel_size=3,
                               stride=3,
                               padding=1)
        self.conv11 = nn.Conv2d(in_depth,
                                depth,
                                kernel_size=3,
                                stride=3,
                                padding=1)
        if not self.first:
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, signal):
        """
            Проход вперед для блока ResNet.

            Параметры:
                signal (torch.Tensor): Входной тензор.

            Возвращает:
                torch.Tensor: Выходной тензор.
        """

        # x is (B x d_in x T)
        prev = signal
        prev_mp = self.conv11(signal)
        if not self.first:
            out = self.pre_bn(signal)
            out = self.lrelu(out)
        else:
            out = signal
        out = self.conv1(signal)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out


class MFCCModel(nn.Module):
    """
    Класс, представляющий MFCC-модель.

    Атрибуты:
        conv1 (nn.Conv2d): Слой свертки 1.
        block1-9 (ResNetBlock): Блоки ResNet 1-9.
        mp (nn.MaxPool2d): Слой максимального пулинга.
        lrelu (nn.LeakyReLU): Функция активации LeakyReLU.
        bn (nn.BatchNorm2d): Слой нормализации пакета.
        dropout (nn.Dropout): Слой исключения.
        logsoftmax (nn.LogSoftmax): Функция активации LogSoftmax.
        fc1 (nn.Linear): Полносвязный слой 1.
        fc2 (nn.Linear): Полносвязный слой 2.
        model_name (str): Название модели.

    Методы:
        forward(signal): Выполняет проход вперед по модели.
    """

    def __init__(self):
        super(MFCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.model_name = 'CNN_model_ResNet'

    def forward(self, signal):
        """
        Проход вперед для модели MFCC.

        Параметры:
            signal (torch.Tensor): Входной тензор.

        Возвращает:
            torch.Tensor: Выходной тензор.
        """

        batch_size = signal.size(0)
        signal = signal.unsqueeze(dim=1)
        out = self.conv1(signal)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


def model_eval():
    """
        Функция для оценки модели.

        Создает экземпляр MFCCModel, загружает веса модели из файла 'Model_weights.bin',
        переводит модель в режим оценки (evaluation), перемещает модель на устройство,
        определенное в CFG.device и возвращает модель.

        Возвращает:
            MFCCModel: Экземпляр модели
    """

    model = MFCCModel()
    module_path = os.path.dirname(__file__)
    weights_path = os.path.join(module_path,  'Model/Model_weights.bin')

    model.load_state_dict(torch.load(weights_path,
                                     map_location=torch.device('cpu')))
    model.eval()
    model.to(CFG.device)
    print('Model Evaluated!')
    return model
