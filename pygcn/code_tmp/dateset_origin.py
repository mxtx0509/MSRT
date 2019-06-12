import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
import glob
from pygcn.buildGraph import build_graph

class SRDataset(data.Dataset):
    def __init__(self, adj_dir, feature_dir,json_dir, file_list,adj_size=40,graph_mode='same'):
        super(SRDataset, self).__init__()

        self.adj_dir = adj_dir
        self.feature_dir = feature_dir
        self.json_dir = json_dir
        self.file_list = file_list
        self.adj_size=adj_size
        
        reader_file = open(file_list)
        lines = reader_file.readlines()
        self.labels = []
        self.files = []
        
        for line in lines:
            line = line.strip().split()
            self.files.append(line[0])
            self.labels.append(int(line[1]))

        
        reader_file.close()

    def __getitem__(self, index):
        # For normalize
        video_path = self.files[index]
        video_id = video_path.split('.')[0]
        
        adj_npy,feature_npy = build_graph(video_id,graph_mode,self.json_dir,self.feature_dir)
        label = self.labels[index]

        adj_tensor = torch.FloatTensor(adj_npy)


        feature_tensor = torch.FloatTensor(feature_npy)
        # print ('=====',feature[1])
        # print ('******',feature_tensor[1])
        # print (a)

        return adj_tensor, feature_tensor,label


    def __len__(self):
        return len(self.files)
