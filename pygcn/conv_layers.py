import math
import torch.nn as nn
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable

class BlockConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, kernel_size,stride, adj_size=40 , bias=False):
        super(BlockConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size
        #num = (adj_size - kernel_size)/stride + 1
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

        self.crop1 = CropConvolution(adj_size,self.weight,start_pos=0, end_pos=20,in_features=in_features,out_features=out_features)
        self.crop2 = CropConvolution(adj_size,self.weight,start_pos=10, end_pos=30,in_features=in_features,out_features=out_features)
        self.crop3 = CropConvolution(adj_size,self.weight,start_pos=20, end_pos=40,in_features=in_features,out_features=out_features)
        
        self.out_mask = Variable(torch.ones(adj_size,out_features).cuda())
        self.out_mask[0:10,:]=1
        self.out_mask[10:30,:]=2
        self.out_mask[30:40,:]=1
        self.out_mask  = 1.0/self.out_mask
        #self.out_mask = nn.parallel.scatter(self.out_mask,[0,1])


        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))########???
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        crop1_fea = self.crop1(input, adj)
        crop2_fea = self.crop2(input, adj)
        crop3_fea = self.crop3(input, adj)

        output = crop1_fea + crop2_fea + crop3_fea

        output = output * self.out_mask#[torch.cuda.current_device()].squeeze(0)
        

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CropConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,adj_size,weight,start_pos, end_pos,in_features,out_features,bias=False):
        super(CropConvolution, self).__init__()
        self.input_mask = Variable(torch.zeros(adj_size,adj_size).cuda())
        self.input_mask[start_pos:end_pos,start_pos:end_pos] =1
        #self.input_mask = nn.parallel.scatter(self.input_mask,[0,1])

        self.weight = weight
        self.bn = nn.BatchNorm1d(out_features * adj_size)
    
    
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # print ('===')
        # print (self.weight.grad)  

        adj_ = adj * self.input_mask            #[torch.cuda.current_device()].squeeze(0) 
        output_ = torch.bmm(adj_, support)
        output = output_.view(output_.size(0),output_.size(1)*output_.size(2))

        output = self.bn(output)
        output = output.view(output_.size(0),output_.size(1),output_.size(2))

        return output

    
class BlockConvolution_1(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, kernel_size,stride, adj_size=40 , bias=False):
        super(BlockConvolution_1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size
        #num = (adj_size - kernel_size)/stride + 1
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
        self.crop4 = CropConvolution(adj_size,self.weight,start_pos=0, end_pos=10,in_features=in_features,out_features=out_features)
        self.crop5 = CropConvolution(adj_size,self.weight,start_pos=10, end_pos=20,in_features=in_features,out_features=out_features)
        self.crop6 = CropConvolution(adj_size,self.weight,start_pos=20, end_pos=30,in_features=in_features,out_features=out_features)
        self.crop7 = CropConvolution(adj_size,self.weight,start_pos=30, end_pos=40,in_features=in_features,out_features=out_features)
        
        
        self.out_mask = Variable(torch.zeros(adj_size,out_features).cuda())
        self.out_mask[:,:]=1
        # self.out_mask = nn.parallel.scatter(self.out_mask,[0,1])
        


        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))########???
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        crop4_fea = self.crop4(input, adj)
        crop5_fea = self.crop5(input, adj)
        crop6_fea = self.crop6(input, adj)
        crop7_fea = self.crop7(input, adj)
        
        output = crop4_fea + crop5_fea + crop6_fea + crop7_fea
        output = output * self.out_mask
        
        # print ('^^^^^',output[1,0:20,:])

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


    