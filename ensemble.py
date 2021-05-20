import argparse
import json
import os
import datetime

import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from model.utils import load_train_df

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--metric', default='mean')
parser.add_argument('--ensemble_name', default='b2vit')

options = parser.parse_args()
CFG = json.load(open(options.config))

ensemble_name = options.ensemble_name
metric = options.metric

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 2,
    'learning_rate': 0.01,
    'max_depth': 4,
    'num_leaves':3,
    'lambda_l2' : 0.3,
    'num_iteration': 1000,
    "min_data_in_leaf":1,
    'verbose': 0
}

logistic_params = {

}

oof_path = [
    "tf_efficientnet_b2_tf_efficientnet_b2_oof.csv",
    # "tf_efficientnet_b1_tf_efficientnet_b1_oof.csv",
    # "tf_efficientnet_b0_ver2_tf_efficientnet_b0_oof.csv",
    "efficientnetv2_rw_s_ver3_efficientnetv2_rw_s_oof.csv",
    "tf_efficientnetv2_s_in21k_tf_efficientnetv2_s_in21k_oof.csv",
    "vit_base_patch16_224_vit_base_patch16_224_oof.csv",#0.96
    # "vit_base_resnet50d_224_ver2_vit_base_resnet50d_224_oof.csv", #0.93
    # "skresnext50_32x4d_skresnext50_32x4d_oof.csv",#0.94
    # "seresnext50_32x4d_seresnext50_32x4d_oof.csv",#0.935
    # "tf_efficientnet_b2_ns_tf_efficientnet_b2_ns_oof.csv",#0.93
    # "tf_efficientnet_b3_ns_tf_efficientnet_b3_ns_oof.csv", #0.92
    # "inception_resnet_v2_inception_resnet_v2_oof.csv"#0.92

]

test_path = [
    "tf_efficientnet_b2_tf_efficientnet_b2_test.csv",
    # "tf_efficientnet_b1_tf_efficientnet_b1_test.csv",
    # "tf_efficientnet_b0_ver2_tf_efficientnet_b0_test.csv",
    "efficientnetv2_rw_s_ver3_efficientnetv2_rw_s_test.csv",
    "tf_efficientnetv2_s_in21k_tf_efficientnetv2_s_in21k_test.csv",
    "vit_base_patch16_224_vit_base_patch16_224_test.csv",#0.96
    # "vit_base_resnet50d_224_ver2_vit_base_resnet50d_224_test.csv", #0.93
    # "skresnext50_32x4d_skresnext50_32x4d_test.csv",#0.94
    # "seresnext50_32x4d_seresnext50_32x4d_test.csv",#0.935
    # "tf_efficientnet_b2_ns_tf_efficientnet_b2_ns_test.csv",#0.93
    # "tf_efficientnet_b3_ns_tf_efficientnet_b3_ns_test.csv", #0.92
    # "inception_resnet_v2_inception_resnet_v2_test.csv"#0.92
]

data_path = "./data/output/"

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/ensemble_{ensemble_name}_{metric}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)


class LightGBM():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        model = lgb.train(
            self.params,lgb_train, 
            valid_sets=lgb_valid,
            num_boost_round=1000,
            early_stopping_rounds=100
            )
        preds_val = model.predict(X_valid, num_iteration=model.best_iteration)
        preds_test = model.predict(X_test, num_iteration=model.best_iteration)
        return preds_val, preds_test

class LogisticWrapper():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds_val = model.predict_proba(X_valid)
        preds_test = model.predict_proba(X_test)
        return preds_val, preds_test

class meanWrapper():
    def __init__(self):
        pass

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        preds_val = np.array(X_valid)
        preds_test = np.array(X_test)
        return preds_val, preds_test

class Identfy():
    def __init__(self):
        pass

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        return np.array(y_valid), np.array(X_test)

def load_mean_df(path, output_label=True):
    oof_df = pd.DataFrame()
    label = pd.DataFrame()
    for p in path:
        count = 0
        if output_label:
            one_df = pd.read_csv(data_path + p).drop(["label"], axis=1)
            one_df = one_df/one_df.iloc[0, :].sum()
            if count < 1:
                oof_df = one_df
                count += 1
                continue
            oof_df = oof_df + one_df
        else:
            one_df = pd.read_csv(data_path + p)
            one_df = one_df/one_df.iloc[0, :].sum()
            if count < 1:
                oof_df = one_df
                count += 1
                continue
            oof_df = oof_df + one_df
        oof_df = oof_df / len(path)
    if output_label:    
        label = pd.read_csv(data_path + p).loc[:, ["label"]]
        return oof_df, label
    else:
        return oof_df

def load_df(path, output_label=True):
    oof_df = pd.DataFrame()
    label = pd.DataFrame()
    for p in path:
        if output_label:
            one_df = pd.read_csv(data_path + p).drop(["label"], axis=1)
            one_df = one_df/one_df.iloc[0, :].sum()
            one_df = one_df.rename(columns=lambda s: s + p)
            oof_df = pd.concat([one_df, oof_df], axis=1)
        else:
            one_df = pd.read_csv(data_path + p)
            one_df = one_df/one_df.iloc[0, :].sum()
            one_df = one_df.rename(columns=lambda s: s + p)
            oof_df = pd.concat([one_df, oof_df], axis=1)
    if output_label:    
        label = pd.read_csv(data_path + p).loc[:, ["label"]]
        return oof_df, label
    else:
        return oof_df


cols = ["0", "1"]
def main():
    if metric == "mean":
        oof_df, oof_label = load_mean_df(oof_path)
        test_df = load_mean_df(test_path, output_label=False)
    else:
        oof_df, oof_label = load_df(oof_path)
        test_df = load_df(test_path, output_label=False)

    train = load_train_df("./data/input/train/")
    # print(f'oof_acc : {(oof_label == )}')
    y_preds = []
    oof_sub = pd.DataFrame(index=[i for i in range(train.shape[0])],columns=cols)
    scores_loss = []
    scores_acc = []
    # folds = StratifiedKFold(n_splits=CFG["fold_num"], shuffle=True, random_state=CFG["seed"]).split(np.arange(oof_df.shape[0]), oof_label.values)
    folds = GroupKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), groups=train.id.values)
    if metric == "lgb":
        model = LightGBM(params)
    elif metric == "logistic":
        model = LogisticWrapper(logistic_params)
    elif metric == "mean":
        model = meanWrapper()
    # model = Identfy()
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_train, X_valid = oof_df.iloc[tr_idx, :], oof_df.iloc[val_idx, :]
        y_train, y_valid = oof_label.iloc[tr_idx], oof_label.iloc[val_idx]
        # print(f'X_valid,shape : {X_valid.shape}, y_valid : {y_valid.shape}')
        # print(f'X_valid,columns : {X_valid.columns}, y_valid.columns : {y_valid.columns}')
        y_pred_valid, y_pred_test = model.train_and_predict(X_train, X_valid, y_train, y_valid, test_df)
        # 結果を保存
        y_preds.append(y_pred_test)
        # print(f'y_valid,shape : {y_valid.shape}, y_pred_valid : {y_pred_valid.shape}')
        # print(f'X_valid[:5] : {X_valid[-50:-1]}, y_pred_valid[:5] : {y_pred_valid[-50:-1]}')
        # print(f'y_valid,label : {y_valid.value_counts("label")}, y_pred_valid.label : {np.argmax(y_pred_valid, axis=1).sum()}')
        # print(f'y_pred_valid.argmax : {np.argmax(y_pred_valid, axis=1)}')
        oof_sub.loc[val_idx, cols] = y_pred_valid
        # スコア
        loss = log_loss(y_valid, y_pred_valid)
        scores_loss.append(loss)
        acc = np.mean(y_valid.iloc[:, 0] == np.argmax(y_pred_valid, axis=1))
        scores_acc.append(acc)
        # loss accの記録
        logger.debug(f"log loss: {loss}")
        logger.debug(f"acc: {acc}")

        loss = sum(scores_loss) / len(scores_loss)
        logger.debug('===CV scores loss===')
        logger.debug(f'scores_loss:{scores_loss}\n mean_loss:{loss}')
        
        acc = sum(scores_acc) / len(scores_acc)
        logger.debug('===CV scores acc===')
        logger.debug(f'scores_acc:{scores_acc}\n mean_acc:{acc}')

    print(f'y_valids : {np.shape(oof_sub)}, y_preds : {np.shape(y_preds)}')
    # foldごとのtest推定値を平均
    tst_preds = np.mean(y_preds, axis=0)
    # print(f'valpreds : {val_preds.shape}')
    # 予測結果を保存
    sub = pd.read_csv("./data/input/submission.csv")
    sub = sub.sort_values('Id').reset_index(drop=True)
    sub['label'] = np.argmax(tst_preds, axis=1)
    logger.debug(sub.value_counts("label"))
    sub.to_csv(f'data/output/submission_ensemble_{ensemble_name}_{metric}.csv', index=False)
    oof_sub["label"] = train["label"]
    oof_sub.to_csv(f'data/output/ensemble_{ensemble_name}_{metric}_oof.csv', index=False)

if __name__ == '__main__':
    main()