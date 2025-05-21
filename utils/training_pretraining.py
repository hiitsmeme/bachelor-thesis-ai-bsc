import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve 


def getPrAucIndividualClass(train_or_val, all_labels, all_preds, writer, epoch):
    # stack label arrays vertically
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    # convert them to torch, for gpu compatibility
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    all_pr_auc = []
    all_roc_auc = []
    # loop over 9 tasks
    for class_idx in range(all_labels.shape[1]):
        mask = ~torch.isnan(all_labels[:, class_idx])   # mask out NaN values
        class_labels = all_labels[:,class_idx][mask]    # get labels for current class
        class_preds = all_preds[:,class_idx][mask]      # get model preds for current class
        roc_auc = roc_auc_score(class_labels, class_preds)
        baseline = class_labels.mean()   # baseline is portion of pos labels  
        precision, recall, _ = precision_recall_curve(class_labels, class_preds)
        pr_auc = auc(recall, precision) - baseline # computes delta auc-pr
        all_pr_auc.append(pr_auc)
        all_roc_auc.append(roc_auc)

        # used for logging to tensorboard
        # writer.add_pr_curve(f"{train_or_val} Precision-Recall Class {class_idx}", class_labels, class_preds, epoch)
        # writer.add_scalar(f"{train_or_val} PR-AUC Delta Class {class_idx}", pr_auc, epoch)
        # writer.add_scalar(f"{train_or_val} ROC-AUC Class {class_idx}", roc_auc, epoch)
        
    # return metrics averaged over tasks and not averaged over tasks
    return np.mean(np.array(all_pr_auc)), np.mean(np.array(all_roc_auc)), np.array(all_pr_auc), np.array(all_roc_auc)


def masked_bce_loss(outputs, targets):
    mask = ~torch.isnan(targets)  # Create a mask where values are NOT NaN
    loss = nn.functional.binary_cross_entropy_with_logits(outputs[mask], targets[mask])  # Compute loss only on valid labels
    return loss


def get_datasets(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    def worker_init_fn(worker_id):
        # Adjust the seed based on the worker id to ensure different seeds for each worker.
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    batch_size = 120
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader