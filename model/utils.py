import os
import random
import pandas as pd
import cv2
import re
import numpy as np
import torch

def seed_everything(seed):
    "seed値を一括指定"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    """
    pathからimageの配列を得る
    """
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        print(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def load_train_df(path, output_label=True):
    train_df = pd.DataFrame()
    base_train_data_path = path
    if output_label:
        train_data_labels = ['0', '1']
        for one_label in train_data_labels:
            one_label_df = pd.DataFrame()
            one_label_paths = os.path.join(base_train_data_path, one_label)
            path_list = os.listdir(one_label_paths)
            _path = ','.join(path_list)
            one_label_df['image_path'] = [os.path.join(one_label_paths, f) for f in path_list]
            one_label_df['label'] = one_label
            pattern = '([0-9]*?)_x([0-9]*?)_y([0-9]*?).png'
            results = re.findall(pattern, _path, re.S)
            id_df = pd.DataFrame(results, columns=["id", "x", "y"])
            one_label_df = pd.concat([one_label_df, id_df], axis=1)
            train_df = pd.concat([train_df, one_label_df])
    else:
        path_list = os.listdir(base_train_data_path)
        _path = ','.join(path_list)
        one_label_df = pd.DataFrame()
        one_label_df['image_path'] = path_list
        pattern = '([0-9]*?)_x([0-9]*?)_y([0-9]*?).png'
        results = re.findall(pattern, _path, re.S)
        id_df = pd.DataFrame(results, columns=["id", "x", "y"])
        one_label_df = pd.concat([one_label_df, id_df], axis=1)
        train_df = pd.concat([train_df, one_label_df])

    train_df = train_df.reset_index(drop=True)
    if output_label:
        label_dic = {"0":0, "1":1}
        train_df["label"]=train_df["label"].map(label_dic)
    return train_df


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, min_epoch=5):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.min_epoch = min_epoch
        
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if self.min_epoch > 0:
                self.min_epoch -= 1
            else:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss