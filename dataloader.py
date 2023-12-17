"""
Dataloader module for loaders prepare
"""
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CFG
from dataset import PyaraSoundDataset
from transforms import *


def prepare_loaders(data):
    audio_names = pd.read_csv(CFG.csv_path)
    audio_names = list(audio_names['path'])
    # audio_names= [item_name.split('.')[0] for item_name in os.listdir(CFG.train_path)] #[os.path.join("./train/",item_name)  for item_name in os.listdir(CFG.train_path)]
    random.shuffle(audio_names)
    audio_names = audio_names[:CFG.num_item_all]
    print(f'Number of items we work with:{len(audio_names)}')
    audio_train_valid, audio_test = train_test_split(audio_names, test_size=0.2, random_state=CFG.seed)
    audio_train, audio_valid = train_test_split(audio_train_valid, test_size=0.25, random_state=CFG.seed)
    # print(type(train_ids))
    # print(valid_ids)
    train_dataset = PyaraSoundDataset(annotations_file=data,
                                      audio_dir=audio_train,
                                      transformation=LFCC_spectrogram,
                                      target_sample_rate=CFG.SAMPLE_RATE,
                                      num_samples=CFG.NUM_SAMPLES,
                                      audio_augmentations=audio_augmentations,
                                      mel_augmentations=spec_augmentations,
                                      lable=True)

    valid_dataset = PyaraSoundDataset(annotations_file=data,
                                      audio_dir=audio_valid,
                                      transformation=LFCC_spectrogram,
                                      target_sample_rate=CFG.SAMPLE_RATE,
                                      num_samples=CFG.NUM_SAMPLES,
                                      audio_augmentations=None,
                                      mel_augmentations=None,
                                      lable=True)

    test_dataset = PyaraSoundDataset(annotations_file=data,
                                     audio_dir=audio_test,
                                     transformation=LFCC_spectrogram,
                                     target_sample_rate=CFG.SAMPLE_RATE,
                                     num_samples=CFG.NUM_SAMPLES,
                                     audio_augmentations=None,
                                     mel_augmentations=None,
                                     lable=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.train_bs,
                              num_workers=0,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_bs,
                              num_workers=0,
                              shuffle=False,
                              pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.valid_bs,
                             num_workers=0,
                             shuffle=False,
                             pin_memory=True)
    return train_loader, valid_loader, test_loader
