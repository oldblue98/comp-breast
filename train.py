# all データで学習する
import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, GroupKFold
from model.utils import EarlyStopping, load_train_df, seed_everything
from model.transform import get_train_transforms, get_valid_transforms
from model.dataloader import prepare_dataloader
from model.model import FlowerImgClassifier
from model.epoch_api import train_one_epoch, valid_one_epoch

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--debug', default=False)
parser.add_argument('--device', default="0")
options = parser.parse_args()
CFG = json.load(open(options.config))
device = torch.device('cuda:{}'.format(options.device))

# logger の設定
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/all_{config_filename}_{CFG["model_arch"]}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

def main():

    logger.debug(CFG)

    train = load_train_df("./data/input/train/")
    seed_everything(CFG['seed'])

    # folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    folds = GroupKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), groups=train.id.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # debug
        if fold > 0 and options.debug:
            break

        if fold < 2:
            break
        
        logger.debug(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        train_loader, val_loader = prepare_dataloader(train, (CFG["img_size_h"], CFG["img_size_w"]), trn_idx, val_idx, data_root='./data/train', train_bs=CFG["train_bs"], valid_bs=CFG["valid_bs"], num_workers=CFG["num_workers"], transform_way=CFG["transform_way"])

        model = FlowerImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

        loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        patience = CFG['patience']
        min_epoch = CFG['min_epoch']
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, CFG['verbose_step'],scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                loss_val, accuracy_val = valid_one_epoch(epoch, model, loss_fn, val_loader, device, CFG['verbose_step'], scheduler=None, schd_loss_update=False)
            
            logger.debug(f'epoch : {epoch}, loss_val : {loss_val:.4f}, accuracy_val = {accuracy_val:.4f}')
            torch.save(model.state_dict(),f'save/all_{config_filename}_{CFG["model_arch"]}_fold_{fold}_{epoch}')

            # early stopping
            if epoch > min_epoch:
                early_stopping(loss_val)
            if early_stopping.early_stop:
                print("Early stopping")
                logger.debug(f'Finished epoch : {epoch}, patience : {patience}')
                break
        del model, optimizer, train_loader, val_loader,  scheduler
        torch.cuda.empty_cache()
        logger.debug("\n")

if __name__ == '__main__':
    main()