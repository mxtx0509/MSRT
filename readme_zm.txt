1、解帧 + 随机抽帧，从每个视频中随机抽取20帧，代码/export/home/zm/dataset/ViSR/ViSR_v1.0/code_need/extract_frames.py + random20.py。目前存放在/export/home/zm/dataset/ViSR/ViSR_v1.0/frame + frame20目录。
2、检测图片中存在的人跟物体，/export/home/zm/test/cvpr2019/pytorch-mask-rcnn/run.sh, 检测完成后会生成json文件,目前存放在目录/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_json目录。
3、crop出图片中的物体并用resnet101来提取特征，/export/home/zm/dataset/ViSR/ViSR_v1.0/code_need/crop.py + resnet_zm.py。特征目前存放在/export/home/zm/dataset/ViSR/ViSR_v1.0/frame20_obj_fea目录。
--------------------------------以上均是针对数据集的操作-----------------------------------
4、建立邻接矩阵：./build_per_obj_graph.py build_person_graph.py 生成目录为：./adj/threshold20目录
5、训练模型：./pygcn/run_B.sh 如果目录有变化，在里面修改
6、测试模型：./pygcn/test.sh 如果目录有变化，在里面修改

环境：python3 pytorch0.3