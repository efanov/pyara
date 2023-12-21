"""Модуль для конфигурации """
import torch


class CFG:
    """
        Класс с основными переменными, которые можно модифицировать.
        activation:
            None: модель будет возвращать предсказание 0 или 1.
            Softmax: модель будет возвращать две вероятности классов.
    """
    JUST_PREDICT = False
    DEBUG = False
    DATASET = 'ASV_600'
    Docker = False
    visualize = False
    FULL_DATA = True
    wandb_on = False
    seed = 101
    model_name = 'CNN'
    info = 'DOCKER_CNN_ASV_600_TEST'
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
    ###############################################
    num_classes = 2
    classes = [0, 1]
    activation = None  # 'softmax'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_w = 'Model_weights.bin'
