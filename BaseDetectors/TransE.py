import logging
import torch
import torch.nn as nn
import params
import pandas as pd
from torch.autograd import Variable

scoredict = {}


class TransE(nn.Module):
    def __init__(self):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(params.total_ent,
                                           params.embedding_dim,
                                           max_norm=1)
        self.rel_embeddings = nn.Embedding(params.total_rel,
                                           params.embedding_dim)

        self.criterion = nn.MarginRankingLoss(params.margin, reduction='sum')

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return torch.norm(h + r - t, p=params.p_norm, dim=1)

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        #print('batch_h:',batch_h)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        score = self.get_score(h, t, r)

        pos_score = score[0:int(len(score) / 2)]

        print('#####################', len(score))
        for i in range(len(score)):
            idtriple = str([
                batch_h.cpu().detach().numpy()[i],
                batch_r.cpu().detach().numpy()[i],
                batch_t.cpu().detach().numpy()[i]
            ])
            scoredict[idtriple] = score.cpu().detach().numpy()[i]
        print('#####################', len(scoredict))
        neg_score = score[int(len(score) / 2):len(score)]

        loss = self.criterion(pos_score.cpu(), neg_score.cpu(),
                              torch.Tensor([-1]))
        return loss, pos_score, neg_score, scoredict