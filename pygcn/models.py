import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
#from pygcn.conv_layers import BlockConvolution
import torch

class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        nhid = 512
        self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        self.gc3 = GraphConvolution(nhid, 256,adj_size)
        self.fc = nn.Linear(256,nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, 0.5, training=self.training) 
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = torch.mean(x,1)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc(x)
        
        return x
