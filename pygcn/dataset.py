import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
import glob

class SRDataset(data.Dataset):
    def __init__(self, adj_dir, feature_dir,txt_dir, file_list,adj_size=40,graph_mode='same',Threshold=0.2,is_train=False):
        super(SRDataset, self).__init__()

        self.adj_dir = adj_dir
        self.feature_dir = feature_dir
        self.txt_dir = txt_dir
        self.file_list = file_list
        self.adj_size=adj_size
        
        reader_file = open(file_list)
        lines = reader_file.readlines()
        self.labels = []
        self.files = []
        
        label_dic = {}
        
        for line in lines:
            line = line.strip().split()
            # if line[1] not in label_dic:
                # label_dic[line[1]] = 0
            # label_dic[line[1]]=label_dic[line[1]] +1
            # if label_dic[line[1]] > 300 and is_train == True:
                # continue
            self.files.append(line[0])
            self.labels.append(int(line[1]))

        
        reader_file.close()

    def __getitem__(self, index):
        # For normalize
        video_path = self.files[index]
        video_id = video_path.split('.')[0]
        
        ################adj build
        adj_npy = np.loadtxt(self.adj_dir+video_id+'.txt')
        adj_npy = self.L_Matrix(adj_npy,self.adj_size)
        
        ################feature build
        feature = np.zeros([self.adj_size, 2048], dtype=np.float32)
        txt_name=self.txt_dir+video_id+'.txt'
        reader = open(txt_name,'r')
        lines = reader.readlines()
        count = 0
        for w in lines:
            w = w.strip()
            path = self.feature_dir+video_id+'/' + w + '.npy'
            try:
                feature[count,:] = np.load(path)
            except Exception as e:
                frame_ , type_ , idx_ = w.split('_')
                idx_ = '%02d'%int(idx_)
                new_ = frame_ + '_' + type_ + '_' + idx_
                path = self.feature_dir+video_id+'/' + new_ + '.npy'
            count = count+1
        
        
        ################tensor build
        adj_tensor = torch.FloatTensor(adj_npy)
        feature_tensor = torch.FloatTensor(feature)
        label = self.labels[index]
        #print ('=====\n',adj_tensor)

        return adj_tensor,feature_tensor,label
        
    def L_Matrix(self,adj_npy,adj_size):

        D =np.zeros((adj_size,adj_size))
        for i in range(adj_size):
            tmp = adj_npy[i,:]
            count = np.sum(tmp==1)
            if count>0:
                number = count ** (-1/2)
                D[i,i] = number

        x = np.matmul(D,adj_npy)
        L = np.matmul(x,D)
        return L

    def __len__(self):
        return len(self.files)
