import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        # self.gc3 = GraphConvolution(nhid, 512)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc = nn.Linear(512,nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = torch.mean(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        
        #print (x.size())
        x = self.fc(x)
        #print ('!!!!',type(x))
        
        return x
