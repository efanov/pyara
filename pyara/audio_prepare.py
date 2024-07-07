"""Модуль, в котором описаны функции для предобработки аудио"""

import librosa
import torch
import torchaudio
from spafe.features import bfcc, cqcc
import numpy as np

from config import CFG


# TODO __all__ во всех файлах чтобы в import попадали только написанные функции

def cut_if_necessary(signal, width=CFG.width):
    """
        Обрезает аудиосигнал до CFG.width выборок.

        Параметры:
            signal (torch.Tensor): Аудиосигнал.
            width (int, опционально): Ширина сигнала после обрезки. По умолчанию CFG.width.

        Возвращает:
            torch.Tensor: Обрезанный аудиосигнал.
    """

    if signal.shape[2] > width:
        signal = signal[:, :, 0:width]
    return signal


def right_pad_if_necessary(signal, width=CFG.width):
    """
     Выполняет дополнение последнего измерения входного сигнала справа, если необходимо.

     Параметры:
     - signal (torch.Tensor): Тензор сигнала, представляющий многомерные данные.
                              Должен быть трехмерным тензором
                              с размерностью [batch_size, num_channels, signal_length].

     Возвращает:
     - torch.Tensor: Дополненный сигнал с выполненным дополнением последнего измерения вправо.

     """
    length_signal = signal.shape[2]
    if length_signal < width:
        num_missing_samples = width - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def prepare_signal(voice_path, 
                   pitch_shift=0,
                   width=CFG.width,
                   sample_rate=CFG.SAMPLE_RATE,
                   spec_type = None):
    """
      Подготавливает аудиосигнал для обработки.

      Аргументы:
      - voice_path (str): Путь к аудиофайлу.
      - spec_type (function): Функция извлечения признаков (MFCC_spec / LFCC_spec / CQCC_spec / BFCC_spec)

      Возвращает:
      - signal (torch.Tensor): Подготовленный сигнал для обработки.

      Шаги подготовки сигнала:
      1. Загрузка аудиофайла с помощью функции `torchaudio.load(voice_path)`.
      Возвращает сигнал `signal` и частоту дискретизации `sr`.
      2. Усреднение сигнала `signal` по первому измерению (каналам),
       используя функцию `signal.mean(dim=0)`.
      3. Добавление размерности пакета (batch) к сигналу `signal`,
      используя функцию `signal.unsqueeze(dim=0)`.
      4. Применение преобразования MFCC (Mel-frequency cepstral coefficients)
      к сигналу `signal` с помощью функции `MFCC_spec(signal)`.
         В результате получается спектрограмма MFCC.
      5. Обрезка спектрограммы `signal`, если необходимо,
      с помощью функции `cut_if_necessary(signal)`.
      6. Дополнение спектрограммы `signal`, если необходимо,
      с помощью функции `right_pad_if_necessary(signal)`.
      7. Повторение спектрограммы `signal` 3 раза по первому измерению (пакету),
       используя функцию `signal.repeat(3, 1, 1)`.
      8. Добавление размерности пакета (batch) к спектрограмме `signal`,
       используя функцию `signal.unsqueeze(dim=0)`.
      9. Перемещение спектрограммы `signal` на устройство (например, GPU),
       указанное в `CFG.device`, с помощью функции `signal.to(CFG.device)`.
      10. Возврат подготовленного сигнала `signal`.
      """
    signal, sample_rate = torchaudio.load(voice_path)
    signal = signal.mean(dim=0)
    signal = signal.unsqueeze(dim=0)
    if pitch_shift != 0:
        signal = librosa.effects.pitch_shift(signal.numpy(), sr=sample_rate, n_steps=pitch_shift)
        signal = torch.from_numpy(signal)

    if spec_type == None:
        signal = MFCC_spec(signal)
    else:
        signal = spec_type(signal)

    signal = cut_if_necessary(signal, width)
    signal = right_pad_if_necessary(signal, width)

    signal = signal.repeat(3, 1, 1)
    signal = signal.unsqueeze(dim=0)
    signal = signal.to(CFG.device)
    print(f'Audio signal prepared!')
    return signal

def prepare_signals(voice_paths,
                    pitch_shift=0,
                    width=CFG.width,
                    sample_rate=CFG.SAMPLE_RATE,
                    spec_type = None):
    """
      Подготавливает несколько аудиосигналов для обработки.

      Аргументы:
      - voice_paths (str): Пути к аудиофайлам.

      Возвращает:
      - signal (torch.Tensor): Список из подготовленных сигналов для обработки.
      - spec_type (function): Функция извлечения признаков (MFCC_spec / LFCC_spec / CQCC_spec / BFCC_spec)

      Шаги подготовки сигнала:
      1. Загрузка аудиофайла с помощью функции `torchaudio.load(voice_path)`.
      Возвращает сигнал `signal` и частоту дискретизации `sr`.
      2. Усреднение сигнала `signal` по первому измерению (каналам),
       используя функцию `signal.mean(dim=0)`.
      3. Добавление размерности пакета (batch) к сигналу `signal`,
      используя функцию `signal.unsqueeze(dim=0)`.
      4. Применение преобразования MFCC (Mel-frequency cepstral coefficients)
      к сигналу `signal` с помощью функции `MFCC_spec(signal)`.
         В результате получается спектрограмма MFCC.
      5. Обрезка спектрограммы `signal`, если необходимо,
      с помощью функции `cut_if_necessary(signal)`.
      6. Дополнение спектрограммы `signal`, если необходимо,
      с помощью функции `right_pad_if_necessary(signal)`.
      7. Повторение спектрограммы `signal` 3 раза по первому измерению (пакету),
       используя функцию `signal.repeat(3, 1, 1)`.
      8. Добавление размерности пакета (batch) к спектрограмме `signal`,
       используя функцию `signal.unsqueeze(dim=0)`.
      9. Перемещение спектрограммы `signal` на устройство (например, GPU),
       указанное в `CFG.device`, с помощью функции `signal.to(CFG.device)`.
      10. Возврат подготовленного сигнала `signal`.
      """
    res = []
    for voice_path in voice_paths:
        # TODO сделать зависимость от args, kwargs
        res.append(prepare_signal(voice_path=voice_path, pitch_shift=pitch_shift, width=width, sample_rate=sample_rate, spec_type=spec_type))
    return res

def prediction(model, signal):
    """
    Функция prediction выполняет предсказание с использованием заданной модели для входного сигнала.

    Параметры:
    - model: Модель, которая будет использоваться для предсказания. Должна быть совместима с PyTorch
     и иметь метод forward для выполнения прямого прохода.
    - signal: Входной сигнал, для которого будет выполнено предсказание.
    Должен быть трехмерным тензором с размерностью [batch_size, num_channels, signal_length].

    Возвращает:
    - Предсказанную метку класса для входного сигнала.
    Возвращается значение 1, если модель предсказывает синтезированный голос, и 0, если модель предсказывает реальный голос.

    Пример использования:
    ```python
    import torch

    # Создать модель
    model = MyModel()

    # Создать входной сигнал
    signal = torch.randn(1, 3, 100)
    # Размерность сигнала: [batch_size=1, num_channels=3, signal_length=100]

    # Выполнить предсказание
    prediction = prediction(model, signal)

    print(prediction)  # Выводит: 1
    ```
    """
    model = model.to(CFG.device)

    signal = signal.squeeze()
    with torch.no_grad():
        output = model(signal)
        soft = torch.nn.Softmax(dim=1)
        output = soft(output)
        out = output.argmax(dim=-1).cpu().numpy()
    if out[0] == 1:
        return (1, round(float(output[0][0]), 2))
    elif out[0] == 0:
        return (0, round(float(output[0][0]), 2))

    return 0


# Audio to MFCC spectrogram
MFCC_spec = torchaudio.transforms.MFCC(
    sample_rate=CFG.SAMPLE_RATE,
    n_mfcc=CFG.mels,
    melkwargs={
        "n_fft": 1024,
        "n_mels": CFG.mels,
        "hop_length": 256,
        "mel_scale": "htk",
        'win_length': 1024,
        'window_fn': torch.hann_window,
        'center': False
    },
)

LFCC_spec = torchaudio.transforms.LFCC(
    sample_rate = CFG.SAMPLE_RATE,
    n_filter = 128,
    f_min = 0.0,
    f_max = None,
    n_lfcc = CFG.mels,
    dct_type = 2,
    norm= 'ortho',
    log_lf = False,
    speckwargs = None,
)

def CQCC_spec(signal):
    cqccs = cqcc.cqcc(signal[0].numpy(),
                      fs = CFG.SAMPLE_RATE,
                      num_ceps = CFG.mels)
    return(torch.from_numpy(np.array([cqccs])))
    
def BFCC_spec(signal):
    bfccs = bfcc.bfcc(signal[0].numpy(),
                      fs = CFG.SAMPLE_RATE,
                      num_ceps=CFG.mels,
                      nfilts = CFG.mels*2)
    return(torch.from_numpy(np.array([bfccs])))

def prediction_multiple(model, signals):
    """
    Функция prediction выполняет предсказание с использованием заданной модели для входного сигнала.

    Параметры:
    - model: Модель, которая будет использоваться для предсказания. Должна быть совместима с PyTorch
     и иметь метод forward для выполнения прямого прохода.
    - signal: Можно подать несколько входных сигналов. Входной сигнал, для которого будет выполнено предсказание.
    Должен быть трехмерным тензором с размерностью [batch_size, num_channels, signal_length].

    Возвращает:
    - Предсказанную метку класса для входного сигнала.
    Возвращается значение 1, если модель предсказывает синтезированный голос, и 0,
     если модель предсказывает реальный голос.

    Пример использования:
    ```python
    import torch

    # Создать модель
    model = MyModel()

    # Создать входной сигнал
    signal = torch.randn(1, 3, 100)
    # Размерность сигнала: [batch_size=1, num_channels=3, signal_length=100]

    # Выполнить предсказание
    prediction = prediction(model, signal)

    print(prediction)  # Выводит: 1
    ```
    """
    res = []
    val = []
    model = model.to(CFG.device)
    for signal in signals:
        signal = signal.squeeze()
        with torch.no_grad():
            output = model(signal)
            soft = torch.nn.Softmax(dim=1)
            output = soft(output)
            out = output.argmax(dim=-1).cpu().numpy()
        if out[0] == 1:
            res.append(1)
            val.append(round(float(output[0][0]), 2))
        elif out[0] == 0:
            res.append(0)
            val.append(round(float(output[0][0]), 2))

    return [res, val]
