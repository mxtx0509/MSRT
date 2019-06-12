import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from pygcn.conv_layers import BlockConvolution
import torch

class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        #self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        self.block1 = BlockConvolution(nfeat, nhid,20,10 ,adj_size)
        self.block2 = BlockConvolution(nhid, nhid,20,10 ,adj_size)
        self.block3 = BlockConvolution(nhid, 512,20,10 ,adj_size)

        #self.gc2 = GraphConvolution(nhid, nhid,adj_size)
        #self.gc3 = GraphConvolution(nhid, 512,adj_size)
        self.fc = nn.Linear(512,nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.block1(x, adj))
        x = F.relu(self.block2(x, adj))
        x = F.relu(self.block3(x, adj))

        x = torch.mean(x,1)
        #print (x.size())
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        #print ('!!!!',type(x))
        
        return x
