import argparse
import json
import numpy as np
import pandas as pd
import os
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score
from model.utils import load_train_df

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_name', default='b0b1b2vit')
parser.add_argument('--metric', default='mean')
parser.add_argument('--th', default=0.40)
parser.add_argument('--knn', default=13)
options = parser.parse_args()
ensemble_name = options.ensemble_name
metric = options.metric

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
handler_file = FileHandler(filename=f'./logs/get_knn_ensemble_{ensemble_name}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

th = float(options.th)
knn = int(options.knn)
logger.debug(f'th : {th}, knn : {knn}')

def get_knn(df_tmp):
    model = NearestNeighbors(n_neighbors = knn)
    model.fit(df_tmp.loc[:, ["x","y"]])
    distances, indices = model.kneighbors(df_tmp.loc[:, ["x", "y"]])
    y_valid = np.asarray(np.array(df_tmp.label)[indices[:, 1:]].mean(axis=1) >= th, dtype="int")
    df_tmp["valid"]  = y_valid
    return df_tmp

def main():
    train_df = load_train_df("./data/input/train/")
    oof_csv = pd.read_csv(f'./data/output/ensemble_{ensemble_name}_{metric}_oof.csv')
    oof_df = train_df.drop("label", axis=1).copy()
    oof_df["label"] = oof_csv.loc[:, ["0", "1"]].idxmax(axis=1)
    oof_df.label = oof_df.label.astype(int)
    oof_df.x = oof_df.x.astype(int)//50
    oof_df.y = oof_df.y.astype(int)//50
    oof_df = oof_df.groupby("id").apply(get_knn)
    oof_df.to_csv(f'data/output/ensemble_{ensemble_name}_{metric}_oof_valid.csv', index=False)

    
    print(f'train.shape : {train_df.shape}, oof.shape : {oof_df.shape}')
    print(f'train.col : {train_df.columns}, oof.col : {oof_df.columns}')
    print(f'train.label == oof.valid : {(train_df.label == oof_df.valid).sum()/len(train_df)}' )
    oof_f1score = f1_score(train_df.label, oof_df.valid)
    oof_acc_score = accuracy_score(train_df.label, oof_df.valid)
    logger.debug(f'oof_f1: {oof_f1score}, oof_acc: {oof_acc_score}')

    test_df = pd.DataFrame()
    df1 = pd.read_csv(f'./data/output/submission_ensemble_{ensemble_name}_{metric}.csv')
    _path = ','.join(df1.Id)
    pattern = '([0-9]*?)_x([0-9]*?)_y([0-9]*?).png'
    results = re.findall(pattern, _path, re.S)
    df2 = pd.DataFrame(results, columns=["id", "x", "y"])
    df3 = pd.concat([df2, df1], axis=1)
    test_df = pd.concat([test_df, df3], axis=0)
    test_df.x = test_df.x.astype(int)//50
    test_df.y = test_df.y.astype(int)//50
    test_df = test_df.groupby("id").apply(get_knn)

    sub = pd.read_csv("./data/input/submission.csv")
    sub = sub.sort_values('Id').reset_index(drop=True)
    sub["label"] = test_df.valid
    sub.to_csv(f'data/output/knn_submission_ensemble_{ensemble_name}_{metric}.csv', index=False)

if __name__ == '__main__':
    main()
    