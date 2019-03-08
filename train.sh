python3.6 trainval_net.py --dataset vg --net vgg16 \
                       --bs 12 --nw 16 \
                       --lr 1e-4 --lr_decay_step 4 \
                       --cuda --use_tfb \
                       --r True --checksession 1 --checkepoch 7 --checkpoint 23987 # resume
                       --o adam

