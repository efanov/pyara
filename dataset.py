import os

import torch
import torchaudio
from torch.utils.data import Dataset
from torch import distributions
from config import CFG
from transforms import probability_augmentetion
import matplotlib.pyplot as plt
from visualizations import *


class PyaraSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 audio_augmentations=None,
                 mel_augmentations=None,
                 lable=False):

        self.annotations = annotations_file
        self.audio_dir = audio_dir
        self.audio_augmentation = audio_augmentations
        self.mel_augmentations = mel_augmentations
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.lable = lable

    def __len__(self):
        return len(self.audio_dir)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)

        signal, sr = torchaudio.load(audio_sample_path)
        if signal.shape == (2, signal.shape[1]):
            signal = signal[0]
            signal = signal.unsqueeze(0)
        if CFG.visualize:
            print(f'This is audio with name: {audio_sample_path} !')
            print(f'AUDIO WITHOUT AUGMENTATIONS')
            visualize_audio(signal, sr=sr)

            mel = self.transformation(signal)
            print(F'MEL SPECTROGRAM WITHOUT AUGMENTATIONS')
            plt.figure(figsize=(20, 5))
            plt.imshow(mel.permute(1, 2, 0).squeeze().log())
            plt.title(f'Log Mel Spectrogram of before augmentations ', fontsize=22)
            plt.xlabel('Time', size=20)
            plt.ylabel('Mels', size=20)
            plt.show()

        if self.audio_augmentation != None:
            if probability_augmentetion(0.5):
                signal = self.audio_augmentation(signal)

            if CFG.visualize:
                print(f'AUDIO AFTER AUGMENTATIONS')
                visualize_audio(signal, sr=CFG.SAMPLE_RATE)

        signal = self.transformation(signal)

        if CFG.visualize:
            mel = signal
            plt.figure(figsize=(20, 5))
            plt.imshow(mel.permute(1, 2, 0).squeeze().log())
            plt.title(f'Log Mel Spectrogram of before augmentations ', fontsize=22)
            plt.xlabel('Time', size=20)
            plt.ylabel('Mels', size=20)
            plt.show()

        if self.mel_augmentations != None:
            signal = self.mel_augmentations(signal)

        # signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)

        signal = self._right_pad_if_necessary(signal)
        # signal = signal.repeat(3, 1, 1)
        # signal = torch.squeeze(signal)
        # signal = self.transformation(signal)
        label = self._get_audio_sample_label(index)

        if CFG.visualize:
            print(f'LABEL = {label}')
            mel = signal
            if int(label) == 1:
                title_lable = 'Fake'
            else:
                title_lable = 'Real'
            plt.figure(figsize=(20, 5))
            plt.imshow(mel.permute(1, 2, 0).squeeze().log())
            plt.title(f'Log Mel Spectrogram of {title_lable} audio after padding and augmentations', fontsize=22)
            plt.xlabel('Time', size=20)
            plt.ylabel('Mels', size=20)
            plt.show()
        # signal = signal.repeat(3,1,1)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[2] > CFG.width:
            signal = signal[:, :, 0:CFG.width]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[2]
        if length_signal < CFG.width:
            num_missing_samples = CFG.width - length_signal
            times_to_add = num_missing_samples // length_signal + 1
            sig_add = signal
            for i in range(times_to_add):
                signal = torch.cat((signal, sig_add), 2)
            if signal.shape[2] > CFG.width:
                signal = signal[:, :, 0:CFG.width]

            # last_dim_padding = (0, num_missing_samples)
            # signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        if CFG.DATASET == 'RUS':
            path = self.audio_dir[index]
            if self.lable == True:
                path = os.path.join(CFG.train_path, path)
            else:
                path = os.path.join(CFG.test_path, path)
        else:
            path = self.audio_dir[index]
            if self.lable == True:

                path = os.path.join(CFG.train_path, path)
            else:

                path = os.path.join(CFG.test_path, path)

        return path

    def _get_audio_sample_label(self, index):
        path = self.audio_dir[index]
        # print(path)
        df = self.annotations
        df = df.loc[lambda df: df['path'] == path]
        num = list(df['fake'])
        return torch.Tensor(num)




