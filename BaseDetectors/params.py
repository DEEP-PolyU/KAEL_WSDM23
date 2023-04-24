import argparse
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir_FB = "./data/FB15k-237"
data_dir_WN = "./data/WN18RR"
data_dir_umls = "./data/umls"
data_dir_kinship = "./data/kinship"
data_dir_NELL = "./data/NELL-995"

dir_emb_ent = "entity2vec.txt"
dir_emb_rel = "relation2vec.txt"

out_folder = "./checkpoints"
log_folder = "./log"

# BiLSTM_Attention
alpha = 0.2
dropout = 0.6

learning_rate = 0.003
gama = 1.0
lam = 0.1
anomaly_ratio = 0  #0.01

# Translation_model
total_ent = 0
total_rel = 0
embedding_dim = 100
margin = 0.1  # 100 100 0.1
p_norm = 2
lr = 0.01  #

num_epochs_trans = 10

kkkkk = 1

num_anomaly_num = 326
