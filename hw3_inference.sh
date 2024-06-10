#!/bin/bash

image_path=$1
output_dir=$2
checkpoint=$3

cd ./yolov5

if [ $checkpoint -eq 0 ]
then
   checkpoint_path="../save_checkpoint0.pt"
elif [ $checkpoint -eq 1 ]
then
   checkpoint_path="../save_checkpoint66.pt"
elif [ $checkpoint -eq 2 ]
then
   checkpoint_path="../save_checkpoint132.pt"
elif [ $checkpoint -eq 3 ]
then
   checkpoint_path="../save_checkpoint199.pt"
elif [ $checkpoint -eq 4 ]
then
   checkpoint_path="../save_checkpoint199.pt"
else
   echo "Invalid checkpoint value"
   exit 1
fi

python detect.py \
    --source $image_path \
    --output_dir $output_dir \
    --weight $checkpoint_path \
    --nosave \
    --save-json

