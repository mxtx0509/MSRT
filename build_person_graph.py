#-*- coding:utf-8 -*-  

import numpy as np
import json 
import os
import glob

from scipy.spatial.distance import pdist
ALPHA = 1
BASELINE = 0.2
NAME = BASELINE*100
BETA = 2
GAMMA = 3
SIZE=40
frame_num = 20
def sample_indices( num_segments,num_frames):
    """
    :param record: VideoRecord
    :return: list
    """

    average_duration = num_frames // num_segments
    if average_duration > 0:
        offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
    elif num_frames > num_segments:
        offsets = np.sort(randint(num_frames , size=num_segments))
    else:
        offsets = np.zeros((num_segments,))
    #return offsets + 1
    return offsets 

def random20(video_id,json_dir):
    files = os.listdir(json_dir + video_id)
    if len(files)==0:
        # print ('null!!!!!!!!!',list_path)
        return None 
    frames_20 = []
    frames = []
    for file in files:  
        frame_id = file.split('.')[0]
        frames.append(frame_id)
    frames.sort()
    offsets = sample_indices(num_segments=15,num_frames=len(frames))
    for idx in offsets: 
        frames_20.append(frames[idx])
    # print ('len(frames)',len(frames),frames)
    # print (offsets,'---',frames_20)
    return frames
    
def calEuclideanDistance(vec1,vec2):  
    #dist = np.linalg.norm(vec1 - vec2)
    dist = pdist(np.vstack([vec1,vec2]),'cosine')
    
    return dist[0]  

def compute_score(score,bbox): 
    
    return score

def compute_distance(bbox1_path,bbox2_path): 
    
    bbox1_fea = np.load(bbox1_path)
    bbox2_fea = np.load(bbox2_path)
    dist = calEuclideanDistance(bbox1_fea,bbox2_fea)
    #print bbox1_path,bbox2_path,dist
    return dist

def find_idx(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

def build_adj(rootdir,number_path,path_name,size):
    jsons_ = glob.glob(list_path+'*.json')
    if len(jsons_)==0:
        print ('null!!!!!!!!!',list_path)
        return None,None
    npy_path = '/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_fea/'+number_path+'/'
    jsons_.sort()
    score_list = []
    bbox_list = []
    name_list = []
    person_num = 0
    count = 0

    for json_name in jsons_:
        #json_name = jsons_
        with open(json_name,'r') as load_f:
            data = json.load(load_f)
            if 'person' not in data.keys():
                # print (json_name,'there is no person in the frame ')
                continue
            person_dic = data['person']
            frame_scores = person_dic['scores']
            frame_bbox = person_dic['bbox']
            frame_name = json_name.split('/')[-1].strip('.json')
            for idx in range(person_dic['num']): 
                score = compute_score(frame_scores[idx],frame_bbox[idx])
                score_list.append(score)
                bbox_name = frame_name +'_person_%02d'%idx ### 0012_person_01 0001视频下 0012帧第二个person bbox feature map
                #print bbox_name
                name_list.append(bbox_name)
            person_num = person_num + person_dic['num']
    # print person_num
    # print score_list 
    # print name_list
    video_id_txt = path_name+number_path+'.txt'
    npy_writer = open(video_id_txt,'w')
    same_adj = np.zeros([size, size], dtype=np.int32)
    diff_adj = np.zeros([size, size], dtype=np.int32)
    if person_num ==0:
        print (list_path,'there is no person in this video')
        npy_writer.close()
        return same_adj,diff_adj

    if  person_num>size:
        score_list,name_list = zip(*sorted(zip(score_list,name_list),reverse=True))
        score_list = list(score_list)
        name_list = list(name_list)
        score_list = score_list[0:size]
        name_list = name_list[0:size]
        # print score_list
        # print name_list
        name_list,score_list = zip(*sorted(zip(name_list,score_list)))
        score_list = list(score_list)
        name_list = list(name_list)
        # print score_list
        # print name_list
        person_num = size
    #name list:000134_person_01
    frame_number = []
    
    
    for frame_ in name_list:
        npy_writer.write(frame_+"\n")
        frame_ = frame_.split('_')[0]
        frame_number.append(frame_)
    npy_writer.close()
    #print frame_number
    for i in range(person_num):
        #print npy_path+name_list[i]+'.npy'
        for j in range(i+1,person_num):
            if i==j:
                same_adj[i][j]=2
                continue
            current_framei = name_list[i].split('_')[0]
            current_framej = name_list[j].split('_')[0]
            if current_framei==current_framej:
                continue
            npy_pathi = npy_path+name_list[i]+'.npy'
            npy_pathj = npy_path+name_list[j]+'.npy'
            distance  = compute_distance(npy_pathi,npy_pathj)
            if distance < BASELINE:
                #min_idx = find_min_frameidx(current_framej,frame_number,npy_pathi,name_list)
                framej_idx = find_idx(current_framej,frame_number)
                framej_dis = []
                for m in framej_idx:
                    npy_pathm = npy_path+name_list[m]+'.npy'
                    framej_dis.append(compute_distance(npy_pathi,npy_pathm))
                min_dis_idx = framej_dis.index(min(framej_dis))
                result = framej_idx[min_dis_idx]
                #print>>writer, list_path.split('/')[-2]+'/'+name_list[i],'\t',list_path.split('/')[-2]+'/'+name_list[result],'\t',framej_dis
                same_adj[i][result] = 1
                break
    for i in range(person_num):
        same_adj[i][i]=1
    same_adj += same_adj.T - np.diag(same_adj.diagonal())
    # np.savetxt('a.txt',same_adj,fmt='%d')

        
    for i in range(person_num):
        #print npy_path+name_list[i]+'.npy'
        for j in range(i+1,person_num):
            current_framei = name_list[i].split('_')[0]
            current_framej = name_list[j].split('_')[0]
            if current_framei==current_framej:
                diff_adj[i][j] = 1
                continue
            framej_idx = find_idx(current_framej,frame_number)
            for m in framej_idx:
                if same_adj[i][m]==0:
                    diff_adj[i][m] = 1
            #print npy_path+name_list[i]+'.npy', framej_idx
            break
    diff_adj += diff_adj.T - np.diag(diff_adj.diagonal())
            
    # np.savetxt('b.txt',diff_adj,fmt='%d')
    return same_adj,diff_adj


rootdir = '/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_json/'
dir_path = os.listdir(rootdir)
dir_path.sort()
count=0


path_npy_same = './adj/frame%d/adj_same/'%frame_num
path_npy_diff = './adj/frame%d/adj_diff/'%frame_num
path_name = './adj/frame%d/person_name/'%frame_num

if not os.path.exists(path_npy_same):
    os.makedirs(path_npy_same)
if not os.path.exists(path_npy_diff):
    os.makedirs(path_npy_diff)    
if not os.path.exists(path_name):
    os.makedirs(path_name)
for number_path in dir_path:
    # print (count)
    # if len(number_path)==4:
        # continue
    print (count,number_path)
    list_path = rootdir + number_path +'/'
    same_adj,diff_adj = build_adj(rootdir,number_path,path_name,size=SIZE)
    #print ('===================')
    if same_adj is None:
        continue
    np.savetxt('%s/%s.txt'%(path_npy_same,number_path),same_adj,fmt='%d')
    np.savetxt('%s/%s.txt'%(path_npy_diff,number_path),diff_adj,fmt='%d')
    #adj_graph[count,:,:]=adj
    count = count+1










