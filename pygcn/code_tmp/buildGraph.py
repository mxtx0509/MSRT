import numpy as np
import json 
import os
import glob
from numpy.random import randint
from scipy.spatial.distance import pdist
ALPHA = 1

BETA = 2
GAMMA = 3
SIZE=40


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

def random20(video_id,json_dir):
    files = os.listdir(json_dir + video_id)
    frames_20 = []
    frames = []
    for file in files:  
        frame_id = file.split('.')[0]
        frames.append(frame_id)
    frames.sort()
    # offsets = sample_indices(num_segments=20,num_frames=len(frames))
    # for idx in offsets: 
        # frames_20.append(frames[idx])
    # print ('len(frames)',len(frames),frames)
    # print (offsets,'---',frames_20)
    return frames
    
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


def build_graph(video_id,graph_mode,json_dir,feature_dir,Threshold):
    frame_20 = random20(video_id,json_dir)
    #print (a)
    if graph_mode == 'same':
        adj,feature = build_same_adj(video_id,frame_20,json_dir,feature_dir,Threshold)
    elif graph_mode == 'diff':
        adj,feature = build_diff_adj(video_id,frame_20,json_dir,feature_dir,Threshold)
    elif graph_mode == 'per_obj':
        adj,feature = build_per_obj_adj(video_id,frame_20,graph_mode,json_dir,feature_dir,Threshold)
    adj = L_Matrix(adj)
    np.savetxt('a.txt',adj,fmt='%.3f')
    return adj,feature


def build_same_adj(video_id,frame_20,json_dir,feature_dir,Threshold):
    json_path = json_dir + video_id + '/'
    score_list = []
    bbox_list = []
    name_list = []
    person_num = 0
    count = 0
    size = 40
    # print (Threshold)
    for frame_name in frame_20:
        json_name = json_path + frame_name + '.json'
        with open(json_name,'r') as load_f:
            data = json.load(load_f)
            if 'person' not in data.keys():
                #print (json_name,'there is no person in the frame ')
                continue
            person_dic = data['person']
            person_scores = person_dic['scores']
            person_bbox = person_dic['bbox']
            for idx in range(person_dic['num']): 
                score = compute_score(person_scores[idx],person_bbox[idx])
                score_list.append(score)
                bbox_name = frame_name +'_person_%02d'%idx ### 0012_person_01 0001视频下 0012 帧第二个 person bbox feature map
                #print bbox_name
                name_list.append(bbox_name)
            person_num = person_num + person_dic['num']

    adj = np.zeros([size, size], dtype=np.int32)
    feature = np.zeros([size, 2048], dtype=np.float32)
    if person_num ==0:
        print (video_id,'there is no person in this video')
        return adj,feature

    if  person_num>size:
        score_list,name_list = zip(*sorted(zip(score_list,name_list),reverse=True))
        score_list = list(score_list)
        name_list = list(name_list)
        score_list = score_list[0:size]
        name_list = name_list[0:size]

        name_list,score_list = zip(*sorted(zip(name_list,score_list)))
        score_list = list(score_list)
        name_list = list(name_list)
        person_num = size
    frame_number = []
    
    #frame_number 每一个节点对应的帧号['000015', '000015', '000020', '000020', '000029', '000029', '000037', '000037', '000042', '000042', '000050', '000050', '000056', '000056', '000065', '000065', '000065', '000072', '000072', '000076', '000083', '000093', '000095', '000107', '000112', '000118', '000118', '000127', '000127', '000130', '000130', '000142', '000142', '000149', '000149', '000149']
    for frame_ in name_list:
        frame_ = frame_.split('_')[0]
        frame_number.append(frame_)

    # print (len(frame_number))
    # print ('===',frame_number)
    #print ('---',name_list)
    # print ('---',frame_20)
    for i in range(person_num):
        for j in range(i+1,person_num):
            current_framei = name_list[i].split('_')[0]
            current_framej = name_list[j].split('_')[0]
            if current_framei==current_framej:
                continue
            npy_pathi = feature_dir + video_id + '/' + name_list[i] + '.npy'
            npy_pathj = feature_dir + video_id + '/' + name_list[j] + '.npy'
            distance  = compute_distance(npy_pathi,npy_pathj)
            if distance < Threshold:
                #min_idx = find_min_frameidx(current_framej,frame_number,npy_pathi,name_list)
                framej_idx = find_idx(current_framej,frame_number)
                framej_dis = []
                for m in framej_idx:
                    npy_pathm = feature_dir + video_id + '/' + name_list[m] + '.npy'
                    framej_dis.append(compute_distance(npy_pathi,npy_pathm))
                min_dis_idx = framej_dis.index(min(framej_dis))
                result = framej_idx[min_dis_idx]
                adj[i][result] = 1
                break
    for i in range(person_num):
        adj[i][i]=1
    
    adj += adj.T - np.diag(adj.diagonal())
    np.savetxt('b.txt',adj,fmt='%d')
    for count in range (len(name_list)):
        feature_name = feature_dir + video_id + '/' + name_list[count] + '.npy'
        feature[count] = np.load(feature_name)
    #print (adj.shape,feature.shape)
    #print (feature[0])
    return adj,feature

def build_diff_adj(video_id,frame_20,json_dir,feature_dir,Threshold):
    json_path = json_dir + video_id + '/'
    print (Threshold)
    score_list = []
    bbox_list = []
    name_list = []
    person_num = 0
    count = 0
    size = 40
    # print (Threshold)
    for frame_name in frame_20:
        json_name = json_path + frame_name + '.json'
        with open(json_name,'r') as load_f:
            data = json.load(load_f)
            if 'person' not in data.keys():
                #print (json_name,'there is no person in the frame ')
                continue
            person_dic = data['person']
            person_scores = person_dic['scores']
            person_bbox = person_dic['bbox']
            for idx in range(person_dic['num']): 
                score = compute_score(person_scores[idx],person_bbox[idx])
                score_list.append(score)
                bbox_name = frame_name +'_person_%02d'%idx ### 0012_person_01 0001视频下 0012 帧第二个 person bbox feature map
                #print bbox_name
                name_list.append(bbox_name)
            person_num = person_num + person_dic['num']

    adj_same = np.zeros([size, size], dtype=np.int32)
    adj_diff = np.zeros([size, size], dtype=np.int32)
    feature = np.zeros([size, 2048], dtype=np.float32)
    if person_num ==0:
        print (video_id,'there is no person in this video')
        return adj_diff,feature

    if  person_num>size:
        score_list,name_list = zip(*sorted(zip(score_list,name_list),reverse=True))
        score_list = list(score_list)
        name_list = list(name_list)
        score_list = score_list[0:size]
        name_list = name_list[0:size]

        name_list,score_list = zip(*sorted(zip(name_list,score_list)))
        score_list = list(score_list)
        name_list = list(name_list)
        person_num = size
    frame_number = []
    
    #frame_number 每一个节点对应的帧号['000015', '000015', '000020', '000020', '000029', '000029', '000037', '000037', '000042', '000042', '000050', '000050', '000056', '000056', '000065', '000065', '000065', '000072', '000072', '000076', '000083', '000093', '000095', '000107', '000112', '000118', '000118', '000127', '000127', '000130', '000130', '000142', '000142', '000149', '000149', '000149']
    for frame_ in name_list:
        frame_ = frame_.split('_')[0]
        frame_number.append(frame_)

    # print (len(frame_number))
    # print ('===',frame_number)
    #print ('---',name_list)
    # print ('---',frame_20)
    for i in range(person_num):
        for j in range(i+1,person_num):
            current_framei = name_list[i].split('_')[0]
            current_framej = name_list[j].split('_')[0]
            if current_framei==current_framej:
                continue
            npy_pathi = feature_dir + video_id + '/' + name_list[i] + '.npy'
            npy_pathj = feature_dir + video_id + '/' + name_list[j] + '.npy'
            distance  = compute_distance(npy_pathi,npy_pathj)
            if distance < Threshold:
                #min_idx = find_min_frameidx(current_framej,frame_number,npy_pathi,name_list)
                framej_idx = find_idx(current_framej,frame_number)
                framej_dis = []
                for m in framej_idx:
                    npy_pathm = feature_dir + video_id + '/' + name_list[m] + '.npy'
                    framej_dis.append(compute_distance(npy_pathi,npy_pathm))
                min_dis_idx = framej_dis.index(min(framej_dis))
                result = framej_idx[min_dis_idx]
                adj_same[i][result] = 1
                break
    for i in range(person_num):
        adj_same[i][i]=1
    
    adj_same += adj_same.T - np.diag(adj_same.diagonal())
    
    for count in range (len(name_list)):
        feature_name = feature_dir + video_id + '/' + name_list[count] + '.npy'
        feature[count] = np.load(feature_name)
    
    for i in range(person_num):
        #print npy_path+name_list[i]+'.npy'
        for j in range(i+1,person_num):
            current_framei = name_list[i].split('_')[0]
            current_framej = name_list[j].split('_')[0]
            if current_framei==current_framej:
                adj_diff[i][j] = 1
                continue
            framej_idx = find_idx(current_framej,frame_number)
            for m in framej_idx:
                if adj_same[i][m]==0:
                    adj_diff[i][m] = 1
            #print npy_path+name_list[i]+'.npy', framej_idx
            break
    np.savetxt('same.txt',adj_same,fmt='%d')
    np.savetxt('diff.txt',adj_diff,fmt='%d')
    
    adj_diff += adj_diff.T - np.diag(adj_diff.diagonal())
    
    
    return adj_diff,feature

def L_Matrix(adj):
    #print (adj.shape[0])
    size = adj.shape[0]
    D =np.zeros((size,size))
    for i in range(size):
        tmp = adj[i,:]
        count = np.sum(tmp==1)
        if count>0:
            number = count ** (-1/2)
            D[i,i] = number
    x = np.matmul(D,adj)
    L = np.matmul(x,D)
    return L


if __name__=='__main__':
    build_graph(video_id='0000',graph_mode='same',json_dir='/export/home/cjw/zm/test/cvpr2019/json_random200/',feature_dir='/export/home/cjw/zm/test/cvpr2019/obj_feature/npy_norm_200/')




































