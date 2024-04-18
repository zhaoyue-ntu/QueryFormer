# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

# %%
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

# %%
data_path = './data/imdb/'

# %%
class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 128
    lr = 0.001
    # epochs = 200
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
args = Args()

import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

# %%
hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1,100)

# %%
encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']
checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

# %%
from model.util import seed_everything
seed_everything()

# %%
model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                 dropout = args.dropout, n_layers = args.n_layers, \
                 use_sample = True, use_hist = True, \
                 pred_hid = args.pred_hid
                )

# %%
_ = model.to(args.device)

# %%
to_predict = 'cost'

# %%
imdb_path = './data/imdb/'
dfs = []  # list to hold DataFrames
# SQ: added
for i in range(2):
#for i in range(18):
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
    df = pd.read_csv(file)
    dfs.append(df)

full_train_df = pd.concat(dfs)

val_dfs = []  # list to hold DataFrames
for i in range(18,20):
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
    df = pd.read_csv(file)
    val_dfs.append(df)

val_df = pd.concat(val_dfs)

# %%
table_sample = get_job_table_sample(imdb_path+'train')

# %%
train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)

# %%
crit = nn.MSELoss()
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)

# %%


# %%


# %%
methods = {
    'get_sample' : get_job_table_sample,
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': args.device,
    'bs': 512,
}

# %%


# %%


# %%
_ = eval_workload('job-light', methods)

# %%
_ = eval_workload('synthetic', methods)

# %%


# %%


# %%


# %%


# %%



