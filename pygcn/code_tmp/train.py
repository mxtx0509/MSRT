from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.dataset import SRDataset
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
from torch.utils.data import DataLoader
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    # Load data
    adj_dir = '/export/home/cjw/zm/test/cvpr2019/adj/adj_same/'
    feature_dir = '/export/home/cjw/zm/test/cvpr2019/extract_obj/npy_normalize/'
    train_loader , test_loader = get_loader(adj_dir, feature_dir)
    
    # Model and optimizer
    model = GCN(nfeat=2048,
                nhid=args.hidden,
                nclass=8,
                dropout=args.dropout)
    
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    model.cuda()
    # if args.cuda:
        # model.cuda()
        # features = features.cuda()
        # adj = adj.cuda()
        # labels = labels.cuda()
        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        # idx_test = idx_test.cuda()
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch,model,train_loader,optimizer,criterion)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # # Testing
    # test()






def get_loader(adj_dir, feature_dir):
    train_list = './list/train_list_label.txt'
    test_list = './list/test_list_label.txt'
    train_set = SRDataset(adj_dir, feature_dir, train_list)
    test_set = SRDataset(adj_dir, feature_dir, test_list)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    return train_loader , test_loader


def train(epoch,model,train_loader,optimizer,criterion):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    for i, (adj_tensor, feature_tensor,target) in enumerate(train_loader):
        adj_tensor = Variable(adj_tensor.cuda())
        feature_tensor = Variable(feature_tensor.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        output = model(feature_tensor, adj_tensor)
        #print ('=======',output.size())
        loss = criterion(output, target_var)
        print ("epoch is %d "%epoch,"\t","loss is :",loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    # loss_train.backward()
    # optimizer.step()

    # if not args.fastmode:
        # # Evaluate validation set performance separately,
        # # deactivates dropout during validation run.
        # model.eval()
        # output = model(features, adj)
        # print ('features.size',features.size())
        # print ('adj.size',adj.size())
        # print ('output.size',output.size())

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
          # 'loss_train: {:.4f}'.format(loss_train.item()),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          # 'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))



if __name__=='__main__':
    main()





