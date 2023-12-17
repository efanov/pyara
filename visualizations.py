import torch
import torchaudio
from torch.utils.data import Dataset
from config import CFG
import matplotlib.pyplot as plt
from IPython import display

def Melspectrogram_visualize(array: torch.Tensor):
    # print(array.shape)
    plt.figure(figsize=(20, 5))
    plt.imshow(array.squeeze().log())
    plt.xlabel('Time', size=20)
    plt.ylabel('Frequency (Hz)', size=20)
    plt.show()


def visualize_audio(wav: torch.Tensor, sr: int = 22050):
    print(f'Sample rate of this audio: {sr}')
    # Average all channels
    if wav.dim() == 2:
        # Any to mono audio convertion
        wav = wav.mean(dim=0)

    plt.figure(figsize=(20, 5))
    plt.plot(wav, alpha=.7, c='green')
    plt.grid()
    plt.xlabel('Time', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.show()

    display.display(display.Audio(wav, rate=sr))