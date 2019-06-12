CUDA_VISIBLE_DEVICES=3 python -u main_B.py --threshold 0.2 \
                                      --graph_mode 'same' \
                                      --val_step 1 \
                                      --print_freq 32 \
                                      --epochs 90 \
                                      --save_dir './checkpoints/zm_test/b_per_obj/' \
                                      --batch_size 32 \
                                      --dropout 0.5 \
                                      --num_class 8 \
                                      --lr 0.01 \
                                      --optim 'adam' \
                                      | tee log/zm_test/b_per_obj.3.log 
