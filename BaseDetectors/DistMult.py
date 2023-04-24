import logging
import torch
import torch.nn as nn

tripledict = {}
scoredict ={}

class DistMult(nn.Module):
    def __init__(self, params):
        super(DistMult, self).__init__()
        self.params = params
        self.ent_embeddings = nn.Embedding(self.params.total_ent, self.params.embedding_dim)
        self.rel_embeddings = nn.Embedding(self.params.total_rel, self.params.embedding_dim)

        # self.criterion = nn.Softplus()
        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return torch.sum(h * t * r, dim=1)

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self.get_score(h, t, r)

        pos_score = score[0: int(len(score) / 2)]
        
        for i in range(len(pos_score)):
            scoredict[str([batch_h.cpu().detach().numpy()[i],batch_r.cpu().detach().numpy()[i],batch_t.cpu().detach().numpy()[i]])] = pos_score.cpu().detach().numpy()[i]

        neg_score = score[int(len(score) / 2): len(score)]

        # regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        # loss = torch.mean(self.criterion(score * y)) + self.params.lmbda * regul
        loss = self.criterion(pos_score.cpu(), neg_score.cpu(), torch.Tensor([-1]))
        
        return loss, pos_score, neg_score, scoredict
