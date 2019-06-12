import argparse
import os, sys
import shutil
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import os.path as osp
from pygcn.dataset import SRDataset
from torch.autograd import Variable
import math
#from pygcn.models import GCN
from pygcn.models import GCN
import torch.optim as optim

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--weights', default='', type=str, metavar='PATH', #model_best.pth.tar
                    help='path to weights (default: none)')

parser.add_argument('--val_step',default=1, type=int,
                    help='val step')
parser.add_argument('--save_dir',default='./checkpoints/threshold_1115/A_same12/', type=str, 
                    help='save_dir')
parser.add_argument('--graph_mode',default='same', type=str,
                    help='mode')
parser.add_argument('--num_gpu', default=1, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--print_freq', default=1, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--test_mode', default=False, type=bool, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--epochs', type=int, default=150,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
                    # help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--num_class', type=int, default=8,
                    help='')
parser.add_argument('--threshold', type=float, default=0.12,
                    help='Number of hidden units.')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8,
                    help='Dropout rate (1 - keep probability).')
                    
best_prec1 = 0

def get_loader(adj_dir, feature_dir,txt_dir,SIZE):
    train_list = '/export/home/zm/dataset/ViSR/ViSR_v1.0/list/lv_train.txt'
    test_list = '/export/home/zm/dataset/ViSR/ViSR_v1.0/list/lv_test.txt'

    train_set = SRDataset(adj_dir,feature_dir=feature_dir,txt_dir=txt_dir, file_list = train_list,adj_size=SIZE,graph_mode=args.graph_mode,Threshold=args.threshold)
    test_set = SRDataset(adj_dir, feature_dir=feature_dir,txt_dir=txt_dir, file_list = test_list,adj_size=SIZE,graph_mode=args.graph_mode,Threshold=args.threshold)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            batch_size=args.batch_size*10, shuffle=False)
    return train_loader , test_loader
# def get_loader(adj_dir, feature_dir,json_dir):
    # train_list = '/export/home/cjw/zm/test/cvpr2019/list_5000/train_split0.txt'
    # test_list = '/export/home/cjw/zm/test/cvpr2019/list_5000/test_split0.txt'

    # train_set = SRDataset(feature_dir=feature_dir,json_dir=json_dir, file_list = train_list,adj_size=SIZE,graph_mode=args.graph_mode,Threshold=args.threshold)
    # test_set = SRDataset( feature_dir=feature_dir,json_dir=json_dir, file_list = test_list,adj_size=SIZE,graph_mode=args.graph_mode,Threshold=args.threshold)
    # train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            # batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            # batch_size=args.batch_size, shuffle=False)
    # return train_loader , test_loader
def main():
    SIZE = 40
    global args, best_prec1
    args = parser.parse_args()
    print (args)
    best_acc = 0

    # Create dataloader
    print ('====> Creating dataloader...')
    if not os.path.exists(args.save_dir) :
        os.makedirs(args.save_dir)
    if args.graph_mode == 'same' :
        adj_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/adj_same/'
        txt_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/person_name/'
    elif args.graph_mode == 'diff':
        adj_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/adj_diff/'
        txt_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/person_name/'
    elif args.graph_mode == 'per_obj':
        adj_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/adj_per_obj/'
        txt_dir = '/export/home/zm/test/cvpr2019/pygcn/adj/threshold20/person_obj_name/'
        SIZE = 60
    else:
        print ('arg.graph_mode input wrong!!!!!')
    feature_dir = '/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_fea/'
    train_loader , test_loader = get_loader(adj_dir, feature_dir,txt_dir,SIZE)
    print ('================================')

    
    # load network
    print ('====> Loading the network...')
    model = GCN(adj_size=SIZE,nfeat=2048,nhid=args.hidden,nclass=args.num_class,dropout=args.dropout)
    print (model)
    print ('adj_dir:',adj_dir)
    print ('txt_dir:',txt_dir)
    print ('feature_dir:',feature_dir)
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    # if args.weights!='':
        # ckpt = torch.load(args.save_dir + args.weights)
        # model.module.load_state_dict(ckpt['state_dict'])
        # print ('!!!load weights success !! path is ',args.weights)
    
    # mkdir_if_missing(args.save_dir)
    if args.weights != '':
        try:
            ckpt = torch.load(args.save_dir+args.weights)
            model.module.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !! path is ',args.weights)
        except Exception as e:
            model_init(args.weights,model)
            
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)
    cudnn.benchmark = True
    if args.test_mode == True: 
        args.batch_size = 1024
        train_loader , test_loader = get_loader(adj_dir, feature_dir,txt_dir,SIZE)
        
        acc = validate(test_loader, model, criterion,1)
        
        return 
    for epoch in range(args.start_epoch, args.epochs + 1):
        # acc = validate(test_loader, model, criterion,epoch)
        adjust_lr(optimizer, epoch)
        train(train_loader, model, criterion,optimizer, epoch)
        if epoch% args.val_step == 0:
            acc = validate(test_loader, model, criterion,epoch)
        else:
            acc = 0
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch,
            }, is_best=is_best, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
        #train_loader , test_loader = get_loader(adj_dir, feature_dir,json_dir)
    args.test_mode = True
    args.batch_size = 1024
    train_loader , test_loader = get_loader(adj_dir, feature_dir,txt_dir,SIZE)
    ckpt = torch.load(args.save_dir+'model_best.pth.tar')
    model.module.load_state_dict(ckpt['state_dict'])
    acc = validate(test_loader, model, criterion,1)
    return

def model_init(weights,model):
    print ('attention!!!!!!! load model fail and go on init!!!')
    ckpt = torch.load(args.save_dir+weights)
    pretrained_dict=ckpt['state_dict']
    model_dict = model.module.state_dict()
    model_pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(model_pre_dict)
    model.module.load_state_dict(model_dict)
    for v ,val in model_pre_dict.items() :
        print ('update',v)

def adjust_lr(optimizer, ep):

    if ep < 30:
        lr = 1e-3 * args.num_gpu 
    elif ep < 60:
        lr = 1e-4 * args.num_gpu
    elif ep < 90:
        lr = 1e-5 * args.num_gpu 
    elif ep < 120:
        lr = 1e-6 * args.num_gpu
    else:
        lr = 1e-7 
    for p in optimizer.param_groups:
        p['lr'] = lr
    print ("lr is ",lr)


def train(train_loader, model, criterion,optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    sub = AverageMeter()
    f1_ma = AverageMeter()
    end = time.time()
    # switch to train mode
    model.train()


    for i, (adj_tensor, feature_tensor,target) in enumerate(train_loader):
        # measure data loading time
        adj_tensor = Variable(adj_tensor.cuda())
        feature_tensor = Variable(feature_tensor.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        
        output = model(feature_tensor, adj_tensor)
        #print ('=======',output.size())
        

        loss = criterion(output, target_var)
        #print (output)
        prec1,prec3 = accuracy(output.data, target, topk=(1,3))
        losses.update(loss.data[0], adj_tensor.size(0))
        top1.update(prec1[0], adj_tensor.size(0))
        top3.update(prec3[0], adj_tensor.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3, lr=optimizer.param_groups[-1]['lr'])))


               

        # val_list_1 = [math.ceil(len(train_loader)/3),math.ceil(len(train_loader)*2/3)]
        # if i in val_list_1:
            # acc = validate(test_loader, model, criterion)
            # save_checkpoint({
                # 'state_dict': model.module.state_dict(),
                # 'epoch': epoch,
            # }, is_best=False,train_batch=i, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
            # model.train()



def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    sub = AverageMeter()

    model.eval()

    end = time.time()
    lated = 0
    val_label = []
    val_pre = []

    for i, (adj_tensor, feature_tensor,target) in enumerate(val_loader):
        # measure data loading time
        adj_tensor = Variable(adj_tensor.cuda())
        feature_tensor = Variable(feature_tensor.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        output = model(feature_tensor, adj_tensor)
        #print ('=======',output.size())
        if args.test_mode ==True:
            output_ = output.data.cpu().numpy()
            path_result = './result_fuse/%d/'%int(args.threshold*100)
            print (path_result)
            if not os.path.exists(path_result):
                os.makedirs(path_result)
            np.savetxt('%s/%s.txt'%(path_result,args.graph_mode),output_)
        
                # compute output
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5, classacc = accuracy(output.data, target, topk=(1,5))
        prec1, prec3 = accuracy(output.data, target, topk=(1,3))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        val_label[lated:lated + batch_size] =target
        val_pre [lated:lated+batch_size] = pred.data.cpu().numpy().tolist()[:]
        lated = lated + batch_size

        losses.update(loss.data[0], adj_tensor.size(0))
        top1.update(prec1[0], adj_tensor.size(0))
        top3.update(prec3[0], adj_tensor.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3)))
    #print(val_label)

    count = [0]*args.num_class
    acc = [0]*args.num_class
    pre_new = []
    # val_label=[0, 7, 6, 5, 3, 6, 5, 0, 4, 2, 7, 2, 3, 0, 5, 6, 4, 6, 7, 6, 6, 3, 7, 5, 3, 6, 6, 0, 3, 6, 7, 1, 1, 6, 4, 7, 6, 1, 0, 4, 6, 0, 2, 4, 7, 4, 6, 6, 6, 4, 3, 7, 6, 6, 6, 1]
    # pre_new = [6, 0, 6, 6, 4, 6, 6, 0, 0, 0, 0, 6, 2, 6, 6, 1, 3, 6, 6, 6, 6, 6, 6, 6, 3, 6, 6, 0, 6, 6, 6, 7, 0, 6, 0, 7, 6, 6, 0, 6, 6, 0, 6, 6, 0, 7, 6, 6, 6, 6, 6, 6, 6, 6, 1, 6]
    for i in val_pre:
        for j in i:
            pre_new.append(j)
    for idx in range(len(val_label)):
        count[val_label[idx]]+=1
        if val_label[idx] == pre_new[idx]:
            acc[val_label[idx]]+=1
    classaccuracys = []
    for i in range(args.num_class):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)
    #print(pre_new)
    # writer = open('result/%d.txt'%epoch,'w')
    # for w in pre_new:
        # writer.write('%d\n'%w)
    # writer.close()
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} classacc {classaccuracys} Loss {loss.avg:.5f}'
          .format(top1=top1, top3=top3,classaccuracys = classaccuracys, loss=losses)))

    return top1.avg



def save_checkpoint(state, is_best,save_dir, filename='checkpoint.pth.tar'):
    fpath = '_'.join((str(state['epoch']), filename))
    fpath = osp.join(save_dir, fpath)
    #mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_dir, 'model_best.pth.tar'))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    count = [0]*args.num_class
    acc = [0]*args.num_class
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  #zhuanzhi
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for idx in range(batch_size):
        count[target[idx]]+=1
        if target[idx] == pred[0][idx]:
            acc[target[idx]]+=1
    res = []
    classaccuracys = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    for i in range(args.num_class):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=='__main__':
    main()
