#!/bin/bash

#Please modify the following roots to yours.
dataset_root=~/BBA-CTA/datasets/fundus
model_root=~/BBA-CTA/models/OPTIC
path_save_log=~/BBA-CTA/OPTIC/logs

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=REFUGE

#Optimizer
optimizer=Adam
lr=0.05

warm_n=5

#Command
cd OPTIC
CUDA_VISIBLE_DEVICES=0 python BBA.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--warm_n $warm_n
