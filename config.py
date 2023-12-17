import torch
from torch import nn
from path_func import *


class CFG:
    """
    Class with main variables, which we can modify
    """
    JUST_PREDICT = False
    DEBUG = False
    DATASET = 'ASV_600'
    Docker = True
    visualize = False
    FULL_DATA = True
    wandb_on = False

    seed = 101
    model_name = 'CNN'
    info = 'DOCKER_CNN_ASV_600'
    archive_name = 'Audio'
    width = 300
    train_bs = 128
    valid_bs = 128
    mels = 80
    lstm_layers = 4
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 48000
    num_item_all = 500 if DEBUG else 611829  # 611829  45235 95000
    num_test = 10 if DEBUG else 301  # 1000
    print_every = 1 if DEBUG else 50  # 500
    epochs = 5 if DEBUG else 40  # 35
    ###############################################
    lr = 0.003
    criterion = nn.CrossEntropyLoss()
    scheduler = None  # 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    num_classes = 2
    classes = [0, 1]
    activation = None  # 'softmax'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = test_path_decide(Docker, DATASET)
    save_path = "./result/"
    train_path = train_path_decide(Docker, DATASET)
    csv_path = csv_path_decide(Docker, DATASET)
    my_test_path = 'C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/test/test_samples.csv'
    best_model_w = f''

    wandb_project_name = "ASV_DOCKER"
    wandb_run_name = f"Layers:{lstm_layers},{model_name}, Epochs: {epochs}, Samples: {num_item_all},BS: {train_bs}, "

