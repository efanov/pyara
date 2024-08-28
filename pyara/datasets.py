import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class PyaraSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 signal_length,
                 device
                ):
        """
        Args:
            audio_dir (string): path to directory with audio files.
            annotations_file (string): path to the .tsv file with filenames and labels.
            transformation: audio spectrogram transformation.
            target_sample_rate: sample rate after audio resampling.
            signal_length: length of signal array after resampling
            device: device to use: cpu or cuda
        """
        self.annotations = pd.read_csv(annotations_file, sep='\t')
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.target_signal_length = signal_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.target_signal_length:
            signal = signal[:, :self.target_signal_length]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.target_signal_length:
            length_addition = self.target_signal_length - length_signal
            last_dim_padding = (0, length_addition)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index]['path'])
        return path

    def _get_audio_sample_label(self, index):
        # 0 if real
        # 1 if fake
        return self.annotations.iloc[index]['fake']