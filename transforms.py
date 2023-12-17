"""
Module of transformations for audio signal
"""
import random

import librosa.effects
import torch
import torchaudio
from torch import distributions
from torch import nn

from config import CFG

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=CFG.SAMPLE_RATE,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=CFG.mels,
    window_fn=torch.hann_window,
    center=False
)

MFCC_spectrogram = torchaudio.transforms.MFCC(
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

LFCC_spectrogram = torchaudio.transforms.LFCC(
    sample_rate=16000,
    n_filter=128,
    f_min=0.0,
    f_max=None,
    n_lfcc=80,
    dct_type=2,
    norm='ortho',
    log_lf=False,
    speckwargs=None,
)


def spec_augmentations(spec):
    if probability_augmentetion(0.05):
        specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(20),
            torchaudio.transforms.TimeMasking(40),
        )
        spec = specaug(spec)
    return spec


def probability_augmentetion(prob):
    return True if random.random() < prob else False


def check_and_reshape_tensor(tensor):
    # Получить размерность тензора
    tensor_shape = torch.Size(tensor.size())

    # Проверить условие, что размерность равна [1, любое число]
    if tensor_shape == torch.Size([1, tensor_shape[-1]]):
        return tensor
    else:
        # Привести тензор к нужной форме
        return torch.reshape(tensor, (1, tensor_shape.numel()))


def audio_augmentations(wav):
    """Function that contains audio augmentations """

    # Gausian Nooise augmentation
    if probability_augmentetion(0.2):
        noiser = distributions.Normal(0, 0.05)
        wav = wav + noiser.sample(wav.size())
        wav = torch.clamp(wav, -1, 1)

    # Time stratching - boost
    if probability_augmentetion(0.25):
        wav = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=random.uniform(1, 2))
        wav = torch.from_numpy(wav)

    # Time stratching - slowing down
    elif probability_augmentetion(0.25):
        wav = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=random.uniform(0.5, 1))
        wav = torch.from_numpy(wav)

    # Volume
    if probability_augmentetion(0.2):
        # quiet
        if probability_augmentetion(0.5):
            valer = torchaudio.transforms.Vol(gain=random.uniform(0.2, 1), gain_type='amplitude')
            wav = valer(wav)
        # Louder
        else:
            valer = torchaudio.transforms.Vol(gain=random.uniform(1, 5), gain_type='amplitude')
            wav = valer(wav)

    # Pitch Shifter
    if probability_augmentetion(0.05):
        # Down
        if probability_augmentetion(0.5):
            wav = librosa.effects.pitch_shift(wav.numpy(), sr=16000, n_steps=-5)
            wav = torch.from_numpy(wav)
        # Up
        else:
            wav = librosa.effects.pitch_shift(wav.numpy(), sr=16000, n_steps=-5)
            wav = torch.from_numpy(wav)
    wav = check_and_reshape_tensor(wav)
    return wav

