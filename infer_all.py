import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import  log_loss

# 引数で debug の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--debug', default=False)
parser.add_argument('--device', default=0)
options = parser.parse_args()
device = torch.device('cuda:{}'.format(options.device))

CFG_list = [
    # "./configs/resnext50_32x4d_ver2.json",
    # "./configs/tf_efficientnet_b0_ver2.json",
    # "./configs/tf_efficientnet_b0_ns.json",
    # "./configs/tf_efficientnet_b1.json",
    # "./configs/tf_efficientnet_b1_ns.json"
    # "./configs/tf_efficientnet_b2.json",
    # "./configs/tf_efficientnet_b3_ver2.json",
    # "./configs/tf_efficientnet_b4_ver2.json",
    # "./configs/tf_efficientnet_b5.json",
    # "./configs/tf_efficientnet_b6.json",
    # "./configs/inception_resnet_v2.json",
    # "./configs/seresnext50_32x4d.json",
    # "./configs/vit_base_patch16_224.json"
    # "./configs/vit_base_resnet50d_224_ver2.json",
    # "./configs/vit_large_patch16_224.json",
    # "./configs/tf_efficientnet_b2_ns.json",
    # "./configs/tf_efficientnet_b3_ns.json",
    # "./configs/skresnext50_32x4d.json",
    # "./configs/efficientnetv2_rw_s_ver2.json",
    "./configs/dm_nfnet_f0_ver2.json",
    "./configs/efficientnetv2_rw_s_ver3.json",
    "./configs/eca_nfnet_l1.json"
]

from model.transform import get_train_transforms, get_valid_transforms, get_inference_transforms
from model.dataset import FlowerDataset
from model.model import FlowerImgClassifier
from model.epoch_api import train_one_epoch, valid_one_epoch, inference_one_epoch
from model.utils import seed_everything, load_train_df

# testのpath取得
test = pd.DataFrame()
base_test_data_path = './data/input/test/test_image/'
test['image_path'] = [os.path.join(base_test_data_path, f) for f in os.listdir(base_test_data_path)]
test = test.sort_values('image_path').reset_index(drop=True)


def infer(CFG, logger):

    train = load_train_df("./data/input/train/")
    seed_everything(CFG['seed'])

    # folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    folds = GroupKFold(n_splits=5).split(np.arange(train.shape[0]), groups=train.id.values)
    tst_preds = []
    val_loss = []
    val_acc = []

    # 行数を揃えた空のデータフレームを作成
    cols = ['0',
            '1']
    oof_df = pd.DataFrame(index=[i for i in range(train.shape[0])],columns=cols)
    y_preds_df = pd.DataFrame(index=[i for i in range(test.shape[0])], columns=cols)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # debug用
        if fold > 0 and options.debug:
            break

        logger.debug(' fold {} started'.format(fold))
        input_shape=(CFG["img_size_h"], CFG["img_size_w"])

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = FlowerDataset(valid_, './data/input/train', transforms=get_inference_transforms(input_shape,CFG["transform_way"]), shape=input_shape, output_label=False)

        test_ds = FlowerDataset(test, './data/input/test', transforms=get_inference_transforms(input_shape,CFG["transform_way"]),shape=input_shape, output_label=False)


        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        model = FlowerImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)

        val_preds = []

        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(torch.load(f'save/all_{config_filename}_{CFG["model_arch"]}_fold_{fold}_{epoch}'))
            logger.debug("epoch:{}".format(epoch))
            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, val_loader, device)]
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, tst_loader, device)]
                    logger.debug("test finished")

        val_preds = np.mean(val_preds, axis=0)
        val_loss.append(log_loss(valid_.label.values, val_preds))
        val_acc.append((valid_.label.values == np.argmax(val_preds, axis=1)).mean())
        oof_df.loc[val_idx, cols] = val_preds
        oof_df.loc[val_idx, "label"] = train.loc[val_idx, "label"]
    logger.debug('validation loss = {:.5f}'.format(np.mean(val_loss)))
    logger.debug('validation accuracy = {:.5f}'.format(np.mean(val_acc)))
    tst_preds = np.mean(tst_preds, axis=0)
    y_preds_df.loc[:, cols] = tst_preds #.reshape(len(tst_preds), -1)

    # 予測値を保存
    oof_df.to_csv(f'data/output/{config_filename}_{CFG["model_arch"]}_oof.csv', index=False)
    y_preds_df.to_csv(f'data/output/{config_filename}_{CFG["model_arch"]}_test.csv', index=False)

    del model
    torch.cuda.empty_cache()
    return np.argmax(tst_preds, axis=1)


if __name__ == '__main__':
    for config_filename in CFG_list:
        with open(config_filename) as f:
            CFG = json.load(f)

        # logger の設定
        from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
        logger = getLogger("logger")    #logger名loggerを取得
        logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
        #handler1を作成
        handler_stream = StreamHandler()
        handler_stream.setLevel(DEBUG)
        handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
        #handler2を作成
        config_filename = os.path.splitext(os.path.basename(config_filename))[0]
        handler_file = FileHandler(filename=f'./logs/all_{os.path.basename(config_filename)}_{CFG["model_arch"]}.log')
        handler_file.setLevel(DEBUG)
        handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
        #loggerに2つのハンドラを設定
        logger.addHandler(handler_stream)
        logger.addHandler(handler_file)
        logger.debug("pred start")
        logger.debug(CFG)
        tst_preds_label_all = infer(CFG, logger)
        print(tst_preds_label_all.shape)
        # 予測結果を保存
        sub = pd.read_csv("./data/input/submission.csv")
        sub['label'] = tst_preds_label_all
        logger.debug(sub.value_counts("label"))
        sub.to_csv(f'data/output/submission_{config_filename}_{CFG["model_arch"]}.csv', index=False)