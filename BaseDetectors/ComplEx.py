import logging
import torch
import torch.nn as nn
import params
import pandas as pd
tripledict = {}
scoredict ={}
idvecdict = {}


# dictionary = {}
# dictpath = './jndata/WN18Pytorch/WN18RR-raw.txt'
# ent2id = dict()
# id2ent = dict()
# rel2id = dict()
# id2rel = dict()
# ent2vec = dict()
# rel2vec = dict()
# all_triples = []
# all_triplestotext = []
# print('reading entities')
# entityid = open('./jndata/WN18Pytorch/entity2id.txt')
# print('storing entity dictionary with id and text')
# for line in entityid.readlines():
#     ent, id = map(str,line.strip('\n').split('\t'))
#     ent2id[ent] = int(id)
#     id2ent[id] = str(ent)
# entityvec = pd.read_csv("./jndata/WN18Pytorch/entity2vec.csv",header=None).values.tolist()
# for i in range(len(entityvec)):
#         for j in range(len(entityvec[i])):
#             entityvec[i][j] = round(float(entityvec[i][j]),10)
            
# print('storing entity dictionary with id and vec')
# entityid = open('./jndata/WN18Pytorch/entity2id.txt')
# for line in entityid.readlines():
#     ent,id = map(str,line.strip('\n').split('\t'))
#     ent = str(entityvec[int(id)]).replace('\t',',')
#     ent2vec[int(id)] = ent
# ########################################
# print('reading relations')
# relationid = open('./jndata/WN18Pytorch/relation2id.txt')
# print('storing relation dictionary with id and text')
# for line in relationid.readlines():
#     rel, id = map(str,line.strip('\n').split('\t'))
#     rel2id[rel] = int(id)
#     id2rel[id] = str(rel)
# relationvec = pd.read_csv("./jndata/WN18Pytorch/relation2vec.csv",header=None).values.tolist()
# for i in range(len(relationvec)):
#         for j in range(len(relationvec[i])):
#             relationvec[i][j] = round(float(relationvec[i][j]),10)
# #print(relationvec)
# print('storing relation dictionary with id and vec')
# relationid = open('./jndata/WN18Pytorch/relation2id.txt')
# for line in relationid.readlines():
#     rel,relid = map(str,line.strip('\n').split('\t'))
#     relvec = str(relationvec[int(relid)]).replace('\t',',')
#     #print(rel)
#     rel2vec[int(relid)] = relvec
# #print(rel2vec) 
# ########################################

# dict = open(dictpath)
# for line in dict.readlines():
#     x,y,z,label = line.strip('\n').split(' ')
#     a = [ent2id[x],rel2id[y],ent2id[z]]
#     dictionary[str(a)] = label
# print('字典长度-----------------------------------------------',len(dictionary))
# print(dictionary)




class ComplEx(nn.Module):
    def __init__(self):
        super(ComplEx, self).__init__()

        self.ent_re_embeddings = nn.Embedding(
            params.total_ent, params.embedding_dim
        )
        self.ent_im_embeddings = nn.Embedding(
            params.total_ent, params.embedding_dim
        )
        self.rel_re_embeddings = nn.Embedding(
            params.total_rel, params.embedding_dim
        )
        self.rel_im_embeddings = nn.Embedding(
            params.total_rel, params.embedding_dim
        )
        # self.criterion = nn.Softplus()
        self.criterion = nn.MarginRankingLoss(params.margin, reduction='sum')
        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def get_score(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            dim=1
        )

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
  
        y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self.get_score(h_re, h_im, t_re, t_im, r_re, r_im)
        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]
        
        print('#####################',len(score))
        for i in range(len(score)):
            idtriple = str([batch_h.cpu().detach().numpy()[i],batch_r.cpu().detach().numpy()[i],batch_t.cpu().detach().numpy()[i]])
            # if idtriple in dictionary.keys():
            scoredict[idtriple] = score.cpu().detach().numpy()[i]
        print('#####################',len(scoredict))        
        

        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(h_im ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(t_im ** 2)
            + torch.mean(r_re ** 2)
            + torch.mean(r_im ** 2)
        )
        # loss = torch.mean(self.criterion(score * y)) + params.lmbda * regul
        loss = self.criterion(pos_score.cpu(), neg_score.cpu(), torch.Tensor([-1]))
        return loss, pos_score, neg_score, scoredict
