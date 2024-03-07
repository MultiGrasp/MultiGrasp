'''
Author: Aiden Li
Date: 2022-05-23 17:54:45
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-07-18 14:38:20
Description: OverfitDistanceField
'''
import torch
from torch import nn

class ODField(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.embd_out_dim = 3
        # self.embedder, self.embd_out_dim = get_embedder(8)
        
        self.dist_enc = nn.ModuleList([
            nn.Linear(self.embd_out_dim, 64), self.relu,
            nn.Linear(64, 64), self.relu,
            nn.Linear(64, 64), self.relu,
            nn.Linear(64, 1)
        ])
        self.grad_enc = nn.ModuleList([
            nn.Linear(self.embd_out_dim, 64), self.relu,
            nn.Linear(64, 64), self.relu,
            nn.Linear(64, 64), self.relu,
            nn.Linear(64, 3), self.tanh
        ])
        
        
        for l in self.dist_enc:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
                nn.init.constant_(l.bias, 0)
        for l in self.grad_enc:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
                nn.init.constant_(l.bias, 0)

    def dist(self, x):
        for l in self.dist_enc:
            x = l(x)
        return x
    
    def grad(self, x):
        for l in self.grad_enc:
            x = l(x)
        return x

    def forward(self, x, range_threshold=0.2):
        # x = self.embedder(x)
        pred_dist = self.dist(x)
        pred_grad = self.grad(x)
        
        dist = x.norm(dim=-1).unsqueeze(-1)
        pred_dist = torch.where(dist < range_threshold, pred_dist, dist)
        
        return torch.cat([pred_dist, pred_grad], dim=-1)

