#!/bin/bash
python train.py --exper_name='log20_5_1' --gpu_id='0' --loss_type='wgan_gp' --n_critic=5 --lam_c=10
