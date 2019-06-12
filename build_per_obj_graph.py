import os
import glob
import json
import numpy as np

def build_obj_adj(rootdir,size):
    dir_path = os.listdir(rootdir)
    dir_path.sort()  
    count = 0
    for number_path in dir_path:
        list_path = rootdir + number_path +'/'   #json file path
        jsons_ = glob.glob(list_path+'*.json')
        if len(jsons_)==0:
            print ('null!!!!!!!!!',list_path)
            continue
        f1 = open('/export/home/zm/test/cvpr2019/pygcn/adj/frame20/person_obj_name/%s.txt'%number_path,'a')
        #npy_path = './extract_obj/npy_normalize/'+list_path.split('/')[-2]+'/'
        jsons_.sort()
        vedio_objs = []
        vedio_pers = []
        #vedio_score_per = []
        #vedio_score_obj = []
        frame_names = []
        for json_name in jsons_:
            with open(json_name,'r') as load_f:
                data = json.load(load_f)
                frame_name = json_name.split('/')[-1].strip('.json')
                frame_names.append(frame_name)
                for key1 in data:
                    if key1 == 'person':
                        frame_dic = data['person']
                        frame_scores = frame_dic['scores']
                        frame_bboxes = frame_dic['bbox']
                        frame_num = frame_dic['num']
                        for idx in range(frame_num):
                            #if idx >=10:
                                #bbox_name = frame_name + '_person_%03d'%idx
                            #else:
                            bbox_name = frame_name + '_person_%02d'%idx
                            vedio_pers.append((frame_bboxes[idx],frame_scores[idx],frame_name,bbox_name))
                            #vedio_score_per.append(frame_scores[idx])
                    else:
                        frame_dic = data[key1]
                        frame_scores = frame_dic['scores']
                        frame_bboxes = frame_dic['bbox']
                        frame_num = frame_dic['num']
                        for idx in range(frame_num):
                            #if idx >=10:
                                #bbox_name = frame_name + '_{}_%03d'.format(key1)%idx
                            #else:
                            bbox_name = frame_name + '_{}_%02d'.format(key1)%idx
                            vedio_objs.append((frame_bboxes[idx],frame_scores[idx],frame_name,bbox_name))
                            #vedio_score_obj.append(frame_scores[idx])


        vedio_pers_news =sorted(vedio_pers,key = lambda vedio_pers:vedio_pers[1],reverse = True)
        vedio_objs_news =sorted(vedio_objs,key = lambda vedio_objs:vedio_objs[1],reverse = True)   #sort by score
        vedio_pers_frames = sorted(vedio_pers_news,key = lambda vedio_pers_news:vedio_pers_news[2])
        vedio_objs_frames = sorted(vedio_objs_news,key = lambda vedio_objs_news:vedio_objs_news[2])  #sort by frame
        #print('-'*40)
        #print(vedio_objs_news)
        #print(len(vedio_objs_news))
        person_in_vedios = vedio_pers_frames[:40]
        object_in_vedios = vedio_objs_frames[:20]
        #print(person_in_vedios)
        #print(object_in_vedios)
        values = [0]*20
        count_obj_dic = dict(zip(frame_names,values))
        count_per_dic = dict(zip(frame_names,values))
        #adj = np.zeros([size, size], dtype=np.int32)
        adj = np.zeros([size, size])
        for a in range(len(object_in_vedios)):
            count_obj_dic[object_in_vedios[a][2]] +=1

        for b in range (len(person_in_vedios)):
            count_per_dic[person_in_vedios[b][2]] +=1
        shape,add_shape,per_txt,obj_txt,x,y=0,0,0,0,0,0
        for f in frame_names:
            num_per = count_per_dic[f]
            num_obj = count_obj_dic[f]
            x = x + num_per
            y = y + num_obj
            add_shape = add_shape + num_obj+num_per
            #print(num_per,num_obj,per_txt,x,obj_txt,y)
            for idx in range(per_txt,x):
                f1.write(person_in_vedios[idx][3])
                f1.write('\n')
            for idx in range(obj_txt,y):
                f1.write(object_in_vedios[idx][3])
                f1.write('\n')
            adj_small = np.zeros([num_obj+num_per,num_obj+num_per])
            for i in range(num_obj+num_per):
                for j in range(num_obj+num_per):
                    if i <num_per and j >= num_per:
                        adj_small[i][j] = 1
                    if i >=num_per and j<num_per:
                        adj_small[i][j] = 1
            adj[shape:add_shape,shape:add_shape]=adj_small
            shape = add_shape
            per_txt = per_txt + num_per
            obj_txt = obj_txt + num_obj
            
        np.savetxt('/export/home/zm/test/cvpr2019/pygcn/adj/frame20/adj_per_obj/%s.txt'%number_path,adj,fmt='%d')
        #np.savetxt('/export/home/cjw/zm/test/cvpr2019/adj/adj_per_obj/%s.txt'%number_path,adj)

    return adj






rootdir = '/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_json/'
adj = build_obj_adj(rootdir,size = 60)



