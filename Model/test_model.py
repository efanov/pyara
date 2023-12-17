"""
Model where described the process of model test
"""

import torch
import wandb
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

from config import CFG
from metrics import AverageMeter, min_tDCF
from sklearn.metrics import matthews_corrcoef


def test_model(test_loader, model):
    print(f'WE ARE TESTING OUT MODEL !')
    print()
    test_accuracy_meter = AverageMeter()
    test_f1_meter = AverageMeter()
    test_EER_meter = AverageMeter()
    all_matrix = [[0, 0], [0, 0]]
    full_vec = []
    full_labels = []
    full_MCC_out = []
    for i, batch in enumerate(tqdm(test_loader)):
        # Move batch to device if device != 'cpu'
        wav = batch[0].to(CFG.device)
        # length = batch['length'].to(device)
        label = batch[1].to(CFG.device)
        label = label.reshape(len(label))
        label = torch.tensor(label).long()
        wav = wav.squeeze()
        # print(wav.shape)
        with torch.no_grad():
            output = model(wav)
            out = output.argmax(dim=-1).cpu().numpy()
            out2 = output.softmax(dim=1)
            out2 = out2.transpose(0, 1)
            out2 = out2[1][:]
            out2 = out2.cpu().numpy()
            labels = label.cpu().numpy()
            full_vec.extend(out2)
            full_labels.extend(labels)
        # print(f'output :{output.argmax(dim=-1)}, label : {label}')

        MCC_out = output.argmax(dim=-1).cpu().numpy()
        full_MCC_out.extend(MCC_out)

        matches = (output.argmax(dim=-1) == label).float().mean()
        f1 = f1_score(output.argmax(dim=-1).cpu(), label.cpu(), average='weighted')
        # print(f'Test:{matches.item()}')

        test_accuracy_meter.update(matches.item(), len(batch[0]))
        # test_EER_meter.update(eer[0],              len(batch[0]))
        test_f1_meter.update(f1, len(batch[0]))
        matrix = confusion_matrix(labels, out, labels=[0, 1])
        all_matrix += matrix
        # print(f'F1 :{f1} , EER: {eer}')
        # print(f'Confusion Matrix of all:{all_matrix}')\

    mDCF = min_tDCF(all_matrix)

    print(f'Confusion Matrix of all:{all_matrix}')
    print(f'minDCF : {mDCF}% ')
    fpr, tpr, _ = metrics.roc_curve(full_labels, full_vec)
    EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    print(f'EER : {EER}%')

    test_auc = metrics.roc_auc_score(full_labels, full_vec)
    MCC = matthews_corrcoef(full_labels, full_MCC_out)
    print(f'MCC : {MCC}')
    print(f'Confusion Matrix of all:{all_matrix}')
    print(f'Accuracy on Test Dataset:{test_accuracy_meter.avg} !')
    # display.clear_output()
    wandb.log({
        "Test Accuracy": test_accuracy_meter.avg,
        "Test F1 score": test_f1_meter.avg,
        "Test EER": EER,
        "Test min-tDCF": mDCF,
        "Test_AUC": test_auc,
        "Test_MCC": MCC})
    # display.clear_output()
