#! /bin/bash
python train.py --dataset $1 --learning_rate $2 --save True --batch_size $3 --seed $4
python test.py --dataset $1_test_rst --my_trained_path ../my_trained/IFNRA_$1/IFNRA_$1.pkl --note BUA_batch$3_lr$2_seed$4