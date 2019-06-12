CUDA_VISIBLE_DEVICES=1 python main.py --threshold 0.2 \
                                      --graph_mode 'per_obj' \
                                      --val_step 1 \
                                      --print_freq 16 \
                                      --epochs 90 \
                                      --save_dir './checkpoints/lv_3000/a_per_obj/' \
                                      --batch_size 32 \
                                      --dropout 0.5 \
                                      --num_class 8 \
                                      | tee log/lv_3000/a_per_obj.log  
