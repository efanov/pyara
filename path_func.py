def train_path_decide(Docker, DATASET):
    if Docker:
        return "/app/data/"
    else:
        if DATASET == 'RUS':
            return "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/rus_dataset/"
        elif DATASET == 'UNITED':
            return "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/united_dataset/"
        elif DATASET == 'ASV_600':
            return "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/ASVspoof2021/clips/"


def test_path_decide(Docker, DATASET):
    if Docker:
        return "/app/data/"
    else:
        return "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/test/wav_ilya/"


def csv_path_decide(Docker, DATASET):
    if Docker:
        if DATASET == 'EQUAL':
            return 'equal_dataset'
        elif DATASET == 'ASV_600':
            return 'valid'
    else:
        if DATASET == 'RUS':
            return 'C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/rus_dataset/main_csv'
        elif DATASET == 'UNITED':
            return 'C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/united_dataset/united_csv.csv'
        elif DATASET == 'ASV_600':
            return 'C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/ASVspoof2021/valid'