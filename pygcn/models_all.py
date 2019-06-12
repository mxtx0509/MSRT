import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from pygcn.conv_layers import BlockConvolution
import torch

class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        nhid1= 512
        nhid2= 256
        self.adj_size = adj_size
        self.gc1 = GraphConvolution(nfeat, nhid1 ,adj_size)
        self.block1 = BlockConvolution(nfeat, nhid1,20,10 ,adj_size)


        self.gc3 = GraphConvolution(nhid1, nhid2,adj_size)
        self.block3 = BlockConvolution(nhid1, nhid2,20,10 ,adj_size)
        self.fc = nn.Linear(nhid2,nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, 0.5, training=self.training) 
        non_local= F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.5, training=self.training) 
        local = F.relu(self.block1(x, adj))
        x =  local + non_local
        
        x = F.dropout(x, 0.5, training=self.training) 
        non_local= F.relu(self.gc3(x, adj))
        x = F.dropout(x, 0.5, training=self.training) 
        local = F.relu(self.block3(x, adj))  
        x = local + non_local

        output = torch.mean(x,1)
        
        # non_local = torch.mean(non_local,1)
        # local = torch.mean(local,1)
        # output = torch.cat((non_local,local),1)
        
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.fc(output)
        #print ('!!!!',type(x))
        
        return output
    
    # def weights_init_classifier(self,m):
        # classname = m.__class__.__name__
        # if classname.find('Linear') != -1:
            # nn.init.normal_(m.weight, std=0.001)
            # nn.init.constant_(m.bias, 0.0)