import os
from threading import Thread
from tqdm import tqdm, trange
from model import MAB
import utils as dp
import logging
import param
import torch
import time

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(param.log_folder, param.path + "_" + str(param.ratio) + "_" + str(param.oppratio) + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    device = param.device
    print(device)

    # Initialize Statistical information for all detectors
    anomalies = [0] * param.k
    remain_length = [0] * param.k
    remain_anomalies = [0] * param.k
    para_lambda4 = [0] * param.k

    armchoice = []

    path = param.path + '/' + 'dictionary.csv'
    topk = param.topk
    iteration = param.topk
    opportunities = param.topk

    ablationrecord = {}
    ablationcount = 0
    ablationrewards = 0

    topk = int(topk * param.ratio)
    iteration = int(iteration * param.ratio)
    opportunities = int(opportunities * param.ratio * param.oppratio)
    dictionary, dimension = dp.dictionary(path)
    features, features_list, labels, anomaly_amount = dp.read_csv(path)

    logging.info('Total Opportunities: %d Triple dimension: %d Anomaly amount: %d.' % (opportunities, dimension, anomaly_amount))

    print('--------------Reading Candidate Sets--------------')
    All_triples = dp.read_all()

    for i in range(param.k):
        All_triples[i] = All_triples[i][:int(topk)]

    for i in range(param.k):
        for j in range(len(All_triples[i])):
            r = dp.rewarding(dictionary, All_triples[i][j])
            anomalies[i] += r

    overlaps = []
    overlaprewards = 0

    overlaps = dp.same_elements4(All_triples[0], All_triples[1], All_triples[2], All_triples[3])
    if len(overlaps) == 0:
        overlaps = dp.same_elements4(All_triples[0], All_triples[1], All_triples[0], All_triples[1])
    print('Overlaps:', len(overlaps))

    opportunities = int(opportunities - len(overlaps))
    print('Oracle Opportunities:', opportunities)

    for i in range(len(overlaps)):
        oreward = dp.rewarding(dictionary, overlaps[i])
        overlaprewards += oreward
        if overlaps[i] in All_triples[0]:
            All_triples[0].remove(overlaps[i])
        if overlaps[i] in All_triples[1]:
            All_triples[1].remove(overlaps[i])
        if overlaps[i] in All_triples[2]:
            All_triples[2].remove(overlaps[i])
        if overlaps[i] in All_triples[3]:
            All_triples[3].remove(overlaps[i])

    ablationcount = len(overlaps)
    ablationrewards = overlaprewards
    ablationrecord[int(ablationcount)] = overlaprewards

    for i in range(param.k):
        remain_length[i] = len(All_triples[i])

    for i in range(param.k):
        for j in range(len(All_triples[i])):
            r = dp.rewarding(dictionary, All_triples[i][j])
            remain_anomalies[i] += r


    logging.info('*********** Start Running **************')
    detected = [] #Initialize an empty list for recording
    θ, ita= dp.init_theta_ita_torch()  # Initialize ita
    model = MAB(opportunities,dictionary,detected,θ,ita)
    para_lambda4 = [torch.rand(1, dtype=float, device=device)] * param.k

    θ, ita, r, Y = model.initialize(overlaps,para_lambda4)
    for i in range(param.k):  # update lambda
        para_lambda4[i] = dp.calculate_lambda(r[i], Y[i])

    oraclerewards, oracleTNR, θ, ita, r, Y = model.train(All_triples)
    for i in range(param.k):  # update lambda
        para_lambda4[i] = dp.calculate_lambda(r[i], Y[i])

    iteration = int(iteration - overlaprewards - oraclerewards)

    finalreward, y, θ, ita = model.application()

    logging.info('❀❀❀❀❀❀❀❀❀❀ ITERATIONS END ❀❀❀❀❀❀❀❀❀❀')
    logging.info('----------------------')
    logging.info('Original anomalies in detectors:', anomalies)
    logging.info('Overlaps:', len(overlaps))
    logging.info('Rewards from overlaps:', overlaprewards)
    logging.info('Remaining length in detectors:', remain_length)
    logging.info('Remaining anomalies in detectors:', remain_anomalies)
    logging.info('Opportunities of Query:', opportunities)
    logging.info('Rewards from Query:', oraclerewards)
    logging.info('Oracle TNR:', round((oraclerewards / opportunities) * 100, 2))
    logging.info('Trained MAB:', iteration)
    logging.info('Trained rewards:', round(finalreward, 2))
    logging.info('Trained TNR:', round(finalreward / iteration * 100, 2), '%')
    logging.info('-----------------------')
    total = finalreward + overlaprewards + oraclerewards
    logging.info('Total Rewards', round(total, 2))
    TNRtotal = dp.calTNR(total, topk)
    logging.info('Total TNR:', TNRtotal)
