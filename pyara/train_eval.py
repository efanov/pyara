import torch
import torchaudio
from torch import nn
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, target in tqdm(data_loader):
        inputs, target = inputs.to(device), target.to(device)

        # calculate loss
        prediction = model(inputs)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    model.train(True)
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


def evaluate_model(model, data_loader, device):
    model.train(False)
    probs, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            predictions = nn.functional.softmax(outputs, dim=1)
            prob_target_class = predictions[:, 1]
            
            probs.extend(prob_target_class.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    print("Finished evaluation")
    return probs, labels


def calculate_metrics(probs, labels):
    preds = (np.array(probs) > 0.5).astype(int)
    
    bal_acc = balanced_accuracy_score(labels, preds, sample_weight=None)
    
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    fpr, tpr, thresholds = roc_curve(labels, probs)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    return bal_acc, precision, recall, eer, f1