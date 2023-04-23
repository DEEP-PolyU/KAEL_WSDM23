import math
import param
import numpy as np
import pandas as pd
import torch
import param

device = param.device

All_triples = []

def read_all():
    transe = []
    transes = open(param.path + '/' + 'transe.txt')
    for line in transes.readlines():
        x = list(map(float, line.strip('\n').split(',')))
        for i in range(len(x)):
            x[i] = round(x[i], 10)
        transe.append(x)

    # distmult = []
    # distmults = open(param.path +'/'+'distmult.txt')
    # #ttms= open('./WN18data/ttm.txt')
    # for line in distmults.readlines():
    #     x = list(map(float, line.strip('\n').split(',')))
    #     for i in range(len(x)):
    #         x[i] = round(x[i],10)
    #     distmult.append(x)

    complex = []
    complexs = open(param.path + '/' + 'complex.txt')
    #ttms= open('./WN18data/ttm.txt')
    for line in complexs.readlines():
        x = list(map(float, line.strip('\n').split(',')))
        for i in range(len(x)):
            x[i] = round(x[i], 10)
        complex.append(x)

    ckrl = []
    ckrls = open(param.path + '/' + 'ckrl.txt')
    for line in ckrls.readlines():
        x = list(map(float, line.strip('\n').split(',')))
        for i in range(len(x)):
            x[i] = round(x[i], 10)
        ckrl.append(x)

    ttm = []
    ttms = open(param.path + '/' + 'kgttm.txt')
    #ttms= open('./WN18data/ttm.txt')
    for line in ttms.readlines():
        x = list(map(float, line.strip('\n').split(',')))
        # for i in range(len(x)):
        #     x[i] = round(x[i],10)
        ttm.append(x)

    All_triples.append(complex)
    All_triples.append(transe)
    All_triples.append(ckrl)
    All_triples.append(ttm)
    return All_triples


def read_csv(datapath):

    features = pd.read_csv(
        datapath,
        #usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300])
        usecols=[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
            71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
            88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150
        ])
    labels = pd.read_csv(datapath, usecols=[0])
    features_list = features.values.tolist()
    for i in range(len(features_list)):
        for j in range(len(features_list[i])):
            features_list[i][j] = round(features_list[i][j], 10)
    anomaly_amount = labels[labels.label == 'anomaly'].shape[0]

    return features, features_list, labels, anomaly_amount


def dictionary(path):
    csv = open(path)
    data = []
    dictionary = dict()
    csv.readline()
    for line in csv.readlines():
        x = list(map(float, line.strip('\n').split(',')[1:]))
        #x = line.strip('\n').split(',')[1:]
        dimention = len(x)
        #print(x)
        #print(dimention)
        dictionary[tuple(x)] = line.strip('\n').split(',')[0]
    return dictionary, dimention


def same_elements4(a, b, c, d):
    length = len(a)
    ab = []
    cd = []
    overlaps = []
    for i in range(length):
        for j in range(length):
            if a[i] == b[j]:
                ab.append(a[i])
                break
    for i in range(length):
        for j in range(length):
            if c[i] == d[j]:
                cd.append(c[i])
                break
    for i in range(len(ab)):
        for j in range(len(cd)):
            if ab[i] == cd[j]:
                overlaps.append(ab[i])
                break
    return overlaps




def rewarding(dictionary, triple):
    #print(triple)
    try:
        if dictionary[tuple(triple)] == 'anomaly':
            reward = 1
        else:
            reward = 0
    except:
        print('error', triple)
        reward = 1
    return reward


def Removeduplicates(targetnumber, triples, detected):
    count = 0
    picked = []
    while count != targetnumber:
        for i in range(len(triples)):
            if triples[i] not in detected:
                picked.append(triples[i])
                detected.append(triples[i])
            count += 1
    return picked, detected


def transposeMatrix(m):
    return list(zip(*m))


def calTNR(Rewards, Topk):
    return str(round((Rewards / Topk) * 100, 3)) + '%'


def calculate_lambda(dictionary, triples):
    X = []
    y = []
    for i in range(len(triples)):
        X.append(triples[i])
        reward = rewarding(dictionary, triples[i])
        if reward == 1:
            y.append(1.0)
        else:
            y.append(0)
    from sklearn import linear_model
    alphas_to_test = np.linspace(0.001, 1, num=100)
    # Crossvalidation, normalize
    model = linear_model.RidgeCV(alphas=alphas_to_test,
                                 normalize=True,
                                 store_cv_values=False,
                                 cv=10)
    model.fit(X, y)
    # Select the regularization coefficient that minimizes the mean square error under cross validation to prevent overfitting
    # Which returns the best lambda, called alpha in model
    return model.alpha_


def calExpectation(xi, θ, ita):
    expectation = 0
    # print(len(xi),len(θ))
    for i in range(len(xi)):
        xi[i] = float(xi[i])
        temp = (xi[i]) * (θ[i])
        expectation += temp
    expectation += ita
    return expectation


def calExpectation_torch(xi, θ, ita):
    xi = list2tensor(xi)
    # θ = list2tensor(θ)
    expectation = torch.dot(xi, θ)
    expectation += ita
    return expectation


def list2tensor(a):
    return torch.tensor(a, dtype=float, device=device)


def θ_update(Y, y, λ):
    #θ = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Y),Y)+λ*np.eye(300)),np.transpose(Y)),y)
    θ = np.dot(
        np.dot(
            np.linalg.inv(np.dot(np.transpose(Y), Y) + λ * np.eye(param.eye)),
            np.transpose(Y)), y)
    return θ


def ita_update(Y, X, λ):
    #ita = 0.1588*math.sqrt(np.dot(np.dot(X,np.linalg.inv(np.dot(np.transpose(Y),Y)+λ*np.eye(60))),np.transpose(X)))
    #ita = param.alpha*math.sqrt(np.dot(np.dot(X,np.linalg.inv(np.dot(np.transpose(Y),Y)+λ*np.eye(300))),np.transpose(X)))
    ita = param.alpha * math.sqrt(
        np.dot(
            np.dot(
                X,
                np.linalg.inv(
                    np.dot(np.transpose(Y), Y) + λ * np.eye(param.eye))),
            np.transpose(X)))

    return ita


###################################################################################
def θ_update_torch(Y, y, λ):
    Y = list2tensor(Y).to(param.device)
    y = list2tensor(y).to(param.device)
    θ = torch.matmul(
        torch.matmul(
            torch.linalg.inv(
                torch.matmul(Y.T, Y) +
                λ * torch.eye(param.eye).to(param.device)), Y.T), y)
    return θ


def ita_update_torch(Y, X, λ):
    Y = list2tensor(Y).to(param.device)
    X = list2tensor(X).to(param.device)
    ita = param.alpha * math.sqrt(
        torch.matmul(
            torch.matmul(
                X,
                torch.linalg.inv(
                    torch.matmul(Y.T, Y) +
                    λ * torch.eye(param.eye).to(param.device))), X.T))

    return ita

def init_theta_torch():
    theta = [torch.rand(1, dtype=float, device=device)] * param.k
    return theta

def init_theta_ita_torch():
    theta = [torch.rand(1, dtype=float, device=device)] * param.k
    ita = [torch.rand(1, dtype=float, device=device)] * param.k
    return theta, ita