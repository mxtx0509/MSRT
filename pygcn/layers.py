import math
import torch.nn as nn
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=40 , bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.bn = nn.BatchNorm2d(self.out_features)
        self.bn = nn.BatchNorm1d(out_features * adj_size)
        #self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))########???
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output_ = torch.bmm(adj, support)
        if self.bias is not None:
            output_ =  output_ + self.bias
        output = output_.view(output_.size(0),output_.size(1)*output_.size(2))
        #print (output.size())
        output = self.bn(output)
        output = output.view(output_.size(0),output_.size(1),output_.size(2))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
