#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='doc'
method='inference'
exp='dinov2_small'
split='366'

config=configs/${dataset}.yaml
labeled_id_path=/u03/thanhnv/2_DocumentSegmentation/train.txt
save_path=/u03/thanhnv/1_Code/VT_UniMatch-V2/exp/doc/supervised
checkpoint_path= /kaggle/input/uni2_model/pytorch/default/1/best.pth

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --checkpoint $checkpoint_path\
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log
