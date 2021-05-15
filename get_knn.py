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
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--th', default=0.49)
parser.add_argument('--knn', default=12)
options = parser.parse_args()
CFG = json.load(open(options.config))
config_filename = os.path.splitext(os.path.basename(options.config))[0]

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
handler_file = FileHandler(filename=f'./logs/inference_{config_filename}_{CFG["model_arch"]}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

th = options.th
knn = options.knn

def get_knn(df_tmp):
    y_label = df_tmp.loc[:, "label"]
    model = NearestNeighbors(n_neighbors = knn)
    model.fit(df_tmp.loc[:, ["x","y"]])
    distances, indices = model.kneighbors(df_tmp.loc[:, ["x", "y"]])
    y_valid = np.asarray(np.array(df_tmp.label)[indices[:, 1:]].mean(axis=1) >= th, dtype="int")
    df_tmp["valid"]  = y_valid
    return df_tmp

def main():
    train_df = load_train_df("./data/input/train/")
    oof_csv = pd.read_csv(f'./data/output/{config_filename}_{CFG["model_arch"]}_oof.csv')
    oof_df = train_df.copy()
    oof_df["oof"] = oof_csv.loc[:, ["0", "1"]].idxmax(axis=1)
    oof_df.label = oof_df.label.astype(int)
    oof_df.oof = oof_df.oof.astype(int)
    oof_df.x = oof_df.x.astype(int)
    oof_df.y = oof_df.y.astype(int)
    oof_df = oof_df.groupby("id").apply(get_knn)
    oof_f1score = f1_score(oof_df.label, oof_df.valid)
    oof_acc_score = accuracy_score(oof_df.label, oof_df.valid)
    logger.debug(f'oof_f1: {oof_f1score}, oof_acc: {oof_acc_score}')

    test_df = pd.DataFrame()
    df1 = pd.read_csv(f'./data/output/submission_{config_filename}_{CFG["model_arch"]}.csv')
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
    sub.to_csv(f'data/output/knn_submission_{config_filename}_{CFG["model_arch"]}.csv', index=False)

if __name__ == '__main__':
    main()
    