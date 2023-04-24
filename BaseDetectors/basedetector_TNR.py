from tracemalloc import start
import numpy as np
from numpy.core.numeric import zeros_like
import params
from dataset import Reader
from create_batch import get_pair_batch_train, toarray_float, toarray, get_batch_baseline, get_batch_baseline_test
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from baselines import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import logging
import datetime
import os
import math

start = datetime.datetime.now()


def base_detector(model_name, ratio, num_num):
    # params.num_anomaly_num = ratio
    params.num_anomaly_num = ratio
    params.kkkkk = num_num
    data_path = params.data_dir_umls
    data_name = "UMLS"
    #data_name = "WN18RR"
    dataset = Reader(data_path, "train", isInjectTopK=True)
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(
            params.log_folder,
            model_name + "_" + data_name + "_" + str(params.num_anomaly_num) +
            "_" + str(params.kkkkk) + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    # global dataset, params
    print("Model name:", model_name)
    logger.info('============ Initialized logger ============')
    logger.info('============================================')
    logging.info('There are %d Triples with %d anomalies in the graph.' %
                 (len(dataset.labels), params.num_anomaly_num))

    params.total_ent = dataset.num_entity
    params.total_rel = dataset.num_relation
    # Model
    if model_name == "DistMult":
        model = DistMult()
    elif model_name == "ComplEx":
        model = ComplEx()
    elif model_name == "TransE":
        model = TransE()
    else:
        logging.info('No such a model!!!')
        exit()
    model = model.to(params.device)

    model_saved_path = model_name + "_" + data_name + "_" + str(
        params.num_anomaly_num) + ".ckpt"

    logging.info(model_saved_path)
    model_saved_path = os.path.join(params.out_folder, model_saved_path)
    # model.load_state_dict(torch.load(os.path.join(params.out_folder, "TransE_FB_model_0.05_2740.ckpt")))
    # criterion = nn.MarginRankingLoss(params.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    all_triples = dataset.train_data
    #print(all_triples)
    labels = dataset.labels

    triplelist = dataset.triplelist

    train_idx = list(range(len(all_triples) // 2))

    # print("length of idx & train data", len(train_idx), len(all_labels))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies /
                               params.batch_size)
    for k in range(params.kkkkk):
        for epoch in range(num_iterations):
            batch_h, batch_t, batch_r, batch_y = get_batch_baseline(
                all_triples, train_idx, epoch)

            batch_h_tensor = torch.LongTensor(batch_h).to(params.device)
            batch_t_tensor = torch.LongTensor(batch_t).to(params.device)
            batch_r_tensor = torch.LongTensor(batch_r).to(params.device)
            #print('batch_h_tensor',batch_h_tensor)

            loss, pos_score, neg_score, scoredict = model(
                batch_h_tensor, batch_t_tensor, batch_r_tensor, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'Epoch %d--%d with loss: %f, positive_loss: %f, negative loss: %f'
                % (k, epoch, loss, pos_score[0].cpu().data,
                   neg_score[0].cpu().data))

            torch.save(model.state_dict(), model_saved_path)
    return scoredict, triplelist


anomalies = [0]
model_name = ['TransE']  #"ComplEx", "DistMult" "TransE"

kk = [50]
for s in range(len(kk)):
    for i in range(len(anomalies)):
        for j in range(len(model_name)):
            scoredict, triplelist = base_detector(model_name[j], anomalies[i],
                                                  kk[s])

end = datetime.datetime.now()

print(start)
print(end)

modelname = 'TransE'
dataset = 'FB'  #UMLS WN FB

outpath = './ranking/' + dataset + '/' + modelname + '.txt'

with open(outpath, 'w') as f:
    print('--------Ranking all triples-------')
    ranking = sorted(scoredict.items(), key=lambda x: x[1], reverse=True)
    print('--------Writing to file-------')
    a = 0
    for i in range(len(ranking)):
        if a <= 16281:
            # print('Correct')
            if ranking[i][0] in triplelist.keys():
                #f.write(str(ranking[i][0]).replace('[','').replace(']','').replace(' ','').replace("'",''))
                f.write(
                    str(triplelist[ranking[i][0]]).replace('[', '').replace(
                        ']', '').replace(' ', '').replace("'", ''))
                f.write(',')
                f.write(str(ranking[i][1]))
                f.write('\n')
                a += 1

##########################################################################################

dictionary = {}
rawpath = './raw/' + dataset + '-raw.txt'
rankingpath = outpath
writepath = './labeled/' + dataset + '/' + modelname + '.txt'

dict = open(rawpath)
for line in dict.readlines():
    x, y, z, label = line.strip('\n').split(' ')
    a = [x, y, z]
    dictionary[str(a)] = label
    #print(dictionary)

print('----------------Writing to the file-------------')
with open(writepath, 'w') as f:
    aaa = open(rankingpath)
    for line in aaa.readlines():
        x, y, z, b = line.strip('\n').split(',')
        if str([x, y, z]) in dictionary.keys():
            a = str([x, y, z]).replace('[', '').replace(']', '').replace(
                ' ', '').replace('"', '').replace("'", '')
            f.write(a)
            f.write(',')
            f.write(dictionary[str([x, y, z])])
            f.write('\n')
