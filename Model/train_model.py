from collections import defaultdict

import torch
from tqdm.auto import tqdm
import wandb
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import copy
from sklearn.metrics import matthews_corrcoef
from config import CFG
from metrics import AverageMeter, min_tDCF


def train_model(model, optimizer, train_loader, valid_loader, criterion, directory):
    storage = defaultdict(list)
    best_accuracy = 0
    for epoch in range(1, CFG.epochs + 1):
        print(f'Epoch: {epoch}/{CFG.epochs}')
        train_loss_meter = AverageMeter()
        model.train()
        ####################################################################################################################
        ################################################## TRAINING ########################################################
        ####################################################################################################################

        for i, batch in enumerate(tqdm(train_loader)):
            # Move batch to device if device != 'cpu'
            wav = batch[0].to(CFG.device)
            # length = batch['length'].to(device)
            label = batch[1].to(CFG.device)
            label = label.reshape(len(label))
            label = torch.tensor(label).long()
            # print(f'label:{label}')
            wav = wav.squeeze()
            # print(wav.shape)
            # mel, mel_length = featurizer(wav, length)
            output = model(wav)
            # print(output.shape)
            # print(f'output :{output.argmax(dim=-1).shape}')

            # print(f'label: {label.shape}')
            loss = criterion(output, label)  # need class probabitities

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_meter.update(loss.item())

        storage['train_loss'].append(train_loss_meter.avg)

        ###################################################################################################################
        #############################################  Validation  ########################################################
        ###################################################################################################################
        full_vec = []
        full_labels = []
        full_MCC_out = []
        validation_loss_meter = AverageMeter()
        validation_accuracy_meter = AverageMeter()
        validation_f1_meter = AverageMeter()
        validation_EER_meter = AverageMeter()
        validation_MinDCF_meter = AverageMeter()
        all_matrix = [[0, 0], [0, 0]]
        model.eval()
        for i, batch in enumerate(tqdm(valid_loader)):
            # Move batch to device if device != 'cpu'
            wav = batch[0].to(CFG.device)
            # length = batch['length'].to(device)
            label = batch[1].to(CFG.device)
            label = label.reshape(len(label))
            label = torch.tensor(label).long()
            wav = wav.squeeze()
            with torch.no_grad():
                output = model(wav)
                loss = criterion(output, label)
                out2 = output.softmax(dim=1)
                out2 = out2.transpose(0, 1)
                out2 = out2[1][:]
                out2 = out2.cpu().numpy()
                out = output.argmax(dim=-1).cpu().numpy()
                labels = label.cpu().numpy()
                full_vec.extend(out2)
                full_labels.extend(labels)
                oouutt = output.argmax(dim=-1).cpu().numpy()
                full_MCC_out.extend(oouutt)
            # print(f'output :{output.argmax(dim=-1)}, label : {label}')

            matches = (output.argmax(dim=-1) == label).float().mean()
            f1 = f1_score(output.argmax(dim=-1).cpu(), label.cpu(), average='weighted')
            # print(f'Vallid:{matches.item()}')

            validation_loss_meter.update(loss.item(), len(batch[0]))
            validation_accuracy_meter.update(matches.item(), len(batch[0]))
            # validation_EER_meter.update(eer[0],              len(batch[0]))
            validation_f1_meter.update(f1, len(batch[0]))

            matrix = confusion_matrix(labels, out, labels=[0, 1])
            all_matrix += matrix
            # print(f'F1 :{f1} , EER: {eer}')
            # print(f'Confusion Matrix of all:{all_matrix}')\
        ######################################## SAVE MODEL ###########################################################################

        if validation_accuracy_meter.avg > best_accuracy:
            print(f'Validation_accuracy improved from {best_accuracy} ---> {validation_accuracy_meter.avg} !')
            PATH = f"{directory}/Best_epoch.bin"
            torch.save(model.state_dict(), PATH)
            best_accuracy = validation_accuracy_meter.avg
        ###############################################################################################################################

        # display.clear_output()
        mDCF = min_tDCF(all_matrix)
        validation_MinDCF_meter.update(mDCF)

        print(f'Confusion Matrix of all:{all_matrix}')
        print(f'minDCF : {mDCF}% ')
        plt.title('ROC CURVE WITH EER')
        fpr, tpr, _ = roc_curve(full_labels, full_vec)
        EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
        print(f'EER : {EER}%')

        auc = roc_auc_score(full_labels, full_vec)
        MCC = matthews_corrcoef(full_labels, full_MCC_out)
        print(f'MCC : {MCC}')
        storage['validation_loss'].append(validation_loss_meter.avg)
        storage['validation_accuracy'].append(validation_accuracy_meter.avg)
        storage['validation_F1_score'].append(validation_f1_meter.avg)
        storage['validation_EER'].append(EER)

        wandb.log({"Train Loss": train_loss_meter.avg,
                   "Valid Loss": validation_loss_meter.avg,
                   "Valid Accuracy": validation_accuracy_meter.avg,
                   "Validation F1 score": validation_f1_meter.avg,
                   "Validation EER": EER,
                   "Valiation min-tDCF": validation_MinDCF_meter.avg,
                   "Validation_AUC": auc,
                   "Validation_MCC": MCC})

    ################### Save Last Epoch ##########################################
    best_model_wts = copy.deepcopy(model.state_dict())
    PATH = f"./{directory}/Last_epoch.bin"
    torch.save(model.state_dict(), PATH)