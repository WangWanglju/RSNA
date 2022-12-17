#%%
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, auc, roc_curve
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup



from dataset import TrainDataset, get_transfos
from config import CFG
from model import define_model
from utils import *



def predict(model, dataset, loss_config, batch_size=64, device="cuda"):
    """
    Torch predict function.

    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    preds = np.empty((0,  model.num_classes))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)

            # Forward
            pred, pred_aux = model(x)

            # Get probabilities
            if loss_config['activation'] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config['activation'] == "softmax":
                pred = pred.softmax(-1)

            preds = np.concatenate([preds, pred.cpu().numpy()])

    return preds



def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds, _ = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.optimizer_config['max_grad_norm'])
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  'allocate momery: {memory:.2f}G'
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0],
                          memory= torch.cuda.max_memory_allocated() / 1024.0**3))
        if CFG.wandb and not CFG.debug:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds, _ = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(torch.sigmoid(y_preds).squeeze().to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'allocate momery: {memory:.2f}G'
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          memory= torch.cuda.max_memory_allocated() / 1024.0**3))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# scheduler
# ====================================================
def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds['cancer'].values
    
    train_dataset = TrainDataset(CFG, train_folds, get_transfos())
    valid_dataset = TrainDataset(CFG, valid_folds, get_transfos())

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.data_config['batch_size'],
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.data_config['val_bs'],
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = define_model(name = CFG.model,
                         num_classes=1,
                         num_classes_aux=0,
                         n_channels=3,
                         pretrained_weights="",
                         pretrained=True)

    model.to(device)

    if CFG.llrd:
        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.CFG.optimizer_config['lr'], 
                                                    decoder_lr=CFG.CFG.optimizer_config['lr'] * 2,
                                                    weight_decay=CFG.weight_decay)
    else:
        optimizer_parameters = model.parameters()

    optimizer = AdamW(optimizer_parameters, lr=CFG.optimizer_config['lr'], eps=CFG.optimizer_config['eps'], betas=CFG.optimizer_config['betas'])
    
    num_train_steps = int(len(train_folds) / CFG.data_config['batch_size'] * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([20.]).to(device)) # RMSELoss(reduction="mean")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.]).to(device))
    
    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score = pfbeta_np(valid_labels, predictions)
        optimal_pf1_score, thresh = optimal_f1(valid_labels, predictions)
        pred = (predictions > 0.5).astype(np.int)
        f1_scores = f1_score(valid_labels, pred)
        fpr, tpr, thresholds = roc_curve(valid_labels, predictions)
        pred_auc = auc(fpr, tpr)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - the pf1: {score:.4f}   f1_score:{f1_scores:.4f}   auc: {pred_auc:.4f}')
        LOGGER.info(f'Epoch {epoch+1} - the optimal pf1: {optimal_pf1_score:.4f}   thresh:{thresh:.4f}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] pf1": score,
                       f"[fold{fold}] f1_score": f1_scores,
                       f"[fold{fold}] auc": pred_auc
                        })
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions,
                        'pf1':best_score},
                        exp_dir+f"/{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(exp_dir+f"/{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds["pred_cancer" ] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

#%%
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SAVE_FOLDER = '../input/rsna-breast-cancer-detection/train_images/'

    train = pd.read_csv("../input/rsna-breast-cancer-detection/train.csv")

    #preprocessing
    OUTPUT = './exp/'
    exp_dir = OUTPUT + f'{CFG.exp}-{CFG.model}'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    LOGGER = get_logger(filename=os.path.join(exp_dir,'train'))
    seed_everything(42)
    # ====================================================
    # CV split
    # ====================================================
    Fold = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['cancer'],groups=train['patient_id'])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    print(train.groupby(['fold', 'cancer']).size())
    
    if CFG.debug:
        train = train.groupby('fold').sample(200).reset_index(drop=True)

        
    train['path'] = SAVE_FOLDER + train["patient_id"].astype(str) + "_" + train["image_id"].astype(str) + ".png"
    
    def get_result(df):
        pf = pfbeta_np(df.cancer, df.pred_cancer, beta=1)
        print(pf)
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.train_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(exp_dir+'/oof_df.pkl')
        
    if CFG.wandb and not CFG.debug:
        wandb.finish()