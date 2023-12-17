import os

import pandas as pd
import torch
from torch import nn

from Model.model import *
from Model.test_model import test_model
from Model.train_model import train_model
from Wandb_functions import wandb_init, wandb_login
from config import CFG
from dataloader import prepare_loaders

# %%
wandb_login()

data = pd.read_csv(CFG.csv_path)  # pd.read_csv(CFG.csv_path, sep = '\\t', header=None)

train_loader, valid_loader, test_loader = prepare_loaders(data)

device = CFG.device
print(f"Device: {device}, Available: {torch.cuda.is_available()}, Pytorch_verion: {torch.__version__}")

if CFG.model_name == 'LSTM':
    model = LSTM().to(CFG.device)
    CFG.wandb_run_name = f"{CFG.info}, Layers:{CFG.lstm_layers}, Epochs: {CFG.epochs}, Samples: {CFG.num_item_all},BS: {CFG.train_bs}, "

elif CFG.model_name == 'CNN':
    model = MFCCModel().to(CFG.device)
    CFG.wandb_run_name = f"LFCC {CFG.model_name},  Epochs: {CFG.epochs}, Samples: {CFG.num_item_all},BS: {CFG.train_bs}, "
else:
    #model = EfficientNet.from_pretrained(model_name)  # , num_classes=3
    pass

    # model._fc = torch.nn.Sequential(
    #     nn.Linear(in_features=1280, out_features=625, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(in_features=625, out_features=256, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(in_features=256, out_features=2, bias=True))
    # CFG.wandb_run_name = f"Efficientnet {CFG.model_name},  Epochs: {CFG.epochs}, Samples: {CFG.num_item_all},BS: {CFG.train_bs}, "


model.eval()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
criterion = nn.CrossEntropyLoss()
directory = f'results/result'
try:
    # shutil.rmtree('./result')
    # local_time = time.ctime().replace(' ', '_').replace(':', '.')
    os.mkdir(directory)
    print('PC DIR CREATED')
except Exception:
    print("DIR NOT CREATED")
    pass

run = wandb_init()
train_model(model, optimizer, train_loader, valid_loader, criterion, directory)
test_model(test_loader, model)
run.finish()

# %%

# %%
