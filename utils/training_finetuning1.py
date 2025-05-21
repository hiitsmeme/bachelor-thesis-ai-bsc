import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve 
import numpy as np
import random
import warnings
import pandas as pd
import json


def make_support_query(X, y, task_idx, k=1, l=None, max_pos_query=5, random_state=None):
    """
    For each task (column) in y, select k positives and k negatives
    for the support set. The rest of the samples go into the query set.

    Returns:
        support_set: np.array with features
        query_set_ np.array with labels
    """
    rng    = np.random.default_rng(random_state)
    labels = y[:, task_idx].cpu()

    # --- SUPPORT SET --- #
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = min(k, len(pos_idx))
    n_neg = min(k, len(neg_idx))
    if len(pos_idx) < k:
        warnings.warn(f"Only {len(pos_idx)} positives available, requested {k}.", UserWarning)
    if len(neg_idx) < k:
        warnings.warn(f"Only {len(neg_idx)} negatives available, requested {k}.", UserWarning)

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos>0 else np.array([],dtype=int)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg>0 else np.array([],dtype=int)

    support_idx = np.concatenate([chosen_pos, chosen_neg])

    # --- LEFTOVERS & QUERY SET --- #
    all_idx  = np.arange(len(labels))
    leftover = np.setdiff1d(all_idx, support_idx, assume_unique=True)

    if l is None:
        # take all leftovers
        query_idx = leftover
    else:
        # split leftovers into positives/negatives
        leftover_pos = leftover[labels[leftover] == 1]
        leftover_neg = leftover[labels[leftover] == 0]

        n_query_pos = min(max_pos_query, len(leftover_pos))
        n_query_neg = min(l - n_query_pos, len(leftover_neg))
        total_avail = len(leftover_pos) + len(leftover_neg)
        if total_avail < l:
            warnings.warn(f"Only {total_avail} leftovers, requested l={l}.", UserWarning)

        q_pos = rng.choice(leftover_pos, size=n_query_pos, replace=False) if n_query_pos>0 else np.array([],dtype=int)
        q_neg = rng.choice(leftover_neg, size=n_query_neg, replace=False) if n_query_neg>0 else np.array([],dtype=int)
        query_idx = np.concatenate([q_pos, q_neg])

    # --- BUILD OUTPUTS --- #
    support_set = (X[support_idx], labels[support_idx])
    query_set   = (X[query_idx],   labels[query_idx])

    return support_set, query_set, support_idx, query_idx



def get_support_query_loaders(X, y, task_idx, k, l, max_pos_query, seed, batch_size=120, num_workers=0):
    """
    Splits X,y into support/query for task `task_idx` with k positives & negatives each,
    then wraps each in a DataLoader with the same style as get_datasets.
    """
    def worker_init_fn(worker_id):
        # ensure different RNG streams per worker
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    # support query sets
    (Xs, ys), (Xq, yq), support_idx, query_idx = make_support_query(X, y, task_idx, k=k, l=l, max_pos_query=max_pos_query, random_state=seed)

    # datasets
    support_dataset = TensorDataset(torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32))
    query_dataset = TensorDataset(torch.tensor(Xq, dtype=torch.float32),torch.tensor(yq, dtype=torch.float32))

    # dataloaders
    support_loader = DataLoader(support_dataset,batch_size=batch_size,shuffle=True,worker_init_fn=worker_init_fn,num_workers=num_workers)
    query_loader = DataLoader(query_dataset,batch_size=batch_size,shuffle=True,worker_init_fn=worker_init_fn,num_workers=num_workers)

    return support_loader, query_loader, support_idx, query_idx


def getPrAucIndividualClassFinetuning(train_or_val, all_labels, all_preds, writer, epoch):
    # 1) flatten the batches into one long axis-0 vector
    labels_cat = np.concatenate(all_labels, axis=0)  # shape (total_samples,) or (total_samples, n_tasks)
    preds_cat  = np.concatenate(all_preds,  axis=0)
    
    # 2) ensure we have a 2-D array, even if n_tasks == 1
    if labels_cat.ndim == 1:
        labels_cat = labels_cat[:, None]  # shape (total_samples, 1)
        preds_cat  = preds_cat[:,  None]

    # 3) convert to torch
    all_labels = torch.tensor(labels_cat)
    all_preds  = torch.tensor(preds_cat)

    # # stack label arrays vertically
    # all_labels = np.vstack(all_labels)
    # all_preds = np.vstack(all_preds)
    # # convert them to torch, for gpu compatibility
    # all_labels = torch.tensor(all_labels)
    # all_preds = torch.tensor(all_preds)
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



def classification_loss(model_outputs, targets):
    mask = ~torch.isnan(targets)  # Create a mask where values are NOT NaN
    classifier_loss = nn.functional.binary_cross_entropy_with_logits(model_outputs[mask], targets[mask])  # Compute loss only on valid labels
    return classifier_loss

def propagation_loss(label_propagation_query_set_outputs, targets):
    mask = ~torch.isnan(targets)  # Create a mask where values are NOT NaN
    label_propagation_query_set_loss = nn.functional.binary_cross_entropy_with_logits(label_propagation_query_set_outputs[mask], targets[mask])  # Compute loss only on valid labels
    return label_propagation_query_set_loss