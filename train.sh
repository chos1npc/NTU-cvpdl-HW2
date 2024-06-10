#!/bin/bash
source_train_data=$1    #/home/mmlab206/CVPDL/HW3/hw3_dataset/org/train
source_val_data=$2  #/home/mmlab206/CVPDL/HW3/hw3_dataset/org/val
target_train_data=$3    #/home/mmlab206/CVPDL/HW3/hw3_dataset/fog/train
target_val_data=$4  #/home/mmlab206/CVPDL/HW3/hw3_dataset/fog/val
checkpoint_save_dir=$5

python txtrans.py \
    --input_dir "$(dirname "$source_train_data")/train.coco.json" \
    --output_dir "./yolotext/org/train"
wait
python txtrans.py \
    --input_dir "$(dirname "$source_val_data")/val.coco.json" \
    --output_dir "./yolotext/org/val"
wait

python txtrans.py \
    --input_dir "$(dirname "$target_val_data")/val.coco.json" \
    --output_dir "./yolotext/fog/val"
wait

mkdir -p ./CUT/datasets/s2t/trainA
mkdir -p ./CUT/datasets/s2t/trainB
mkdir -p ./CUT/datasets/s2t/testA
mkdir -p ./CUT/datasets/s2t/testB

mkdir -p ./CUT/datasets/t2s/trainA
mkdir -p ./CUT/datasets/t2s/trainB
mkdir -p ./CUT/datasets/t2s/testA
mkdir -p ./CUT/datasets/t2s/testB

cp -r "$source_train_data"/* ./CUT/datasets/s2t/trainA
cp -r "$source_val_data"/* ./CUT/datasets/s2t/testA
cp -r "$target_train_data"/* ./CUT/datasets/s2t/trainB
cp -r "$target_val_data"/* ./CUT/datasets/s2t/testB

cp -r "$source_train_data"/* ./CUT/datasets/t2s/trainB
cp -r "$source_val_data"/* ./CUT/datasets/t2s/testB
cp -r "$target_train_data"/* ./CUT/datasets/t2s/trainA
cp -r "$target_val_data"/* ./CUT/datasets/t2s/testA

cd ./CUT

python train.py
    --dataroot ./datasets/s2t \
    --name cvpdl_s2t \
    --CUT_mode CUT \
    --display_id -1
wait
python train.py
    --dataroot ./datasets/t2s \
    --name cvpdl_t2s \
    --CUT_mode CUT \
    --display_id -1
wait

python test.py \
-dataroot ./datasets/s2t \
--name cvpdl_s2t \
--model cut \
--results_dir ./test_img/s2t \
--checkpoints_dir ./checkpoints \
--num_test 300 \
--preprocess scale_shortside \
--load_size 320
wait


mv ./datasets/s2t/trainA ./datasets/s2t/tmp
mv ./datasets/s2t/testA ./datasets/s2t/trainA
mv ./datasets/s2t/tmp ./datasets/s2t/testA

python test.py \
-dataroot ./datasets/s2t \
--name cvpdl_s2t \
--model cut \
--results_dir ./test_img/s2t \
--checkpoints_dir ./checkpoints \
--num_test 2575 \
--preprocess scale_shortside \
--load_size 320
wait

python test.py \
-dataroot ./datasets/t2s \
--name cvpdl_s2t \
--model cut \
--results_dir ./test_img/t2s \
--checkpoints_dir ./checkpoints \
--num_test 300 \
--preprocess scale_shortside \
--load_size 320
wait

mv ./datasets/t2s/trainA ./datasets/t2s/tmp
mv ./datasets/t2s/testA ./datasets/t2s/trainA
mv ./datasets/t2s/tmp ./datasets/t2s/testA

python test.py \
-dataroot ./datasets/t2s \
--name cvpdl_s2t \
--model cut \
--results_dir ./test_img/t2s \
--checkpoints_dir ./checkpoints \
--num_test 2575 \
--preprocess scale_shortside \
--load_size 320
wait

cd ../yolov5
python detecttxt.py \
--weights ./yolov5_trainbest.pt \
--source $target_train_data \
--save-txt \
--save-conf \
--conf-thres=0.001 \
--iou-thres=0.6
wait

cd ..
mkdir -p SSDA_dataset/source_data/images/train
mkdir -p SSDA_dataset/source_data/images/train
mkdir -p SSDA_dataset/source_data/labels/val
mkdir -p SSDA_dataset/source_data/labels/val

mkdir -p SSDA_dataset/source_data_fake/images/train
mkdir -p SSDA_dataset/source_data_fake/images/val
mkdir -p SSDA_dataset/source_data_fake/labels/train
mkdir -p SSDA_dataset/source_data_fake/labels/val
cd ./CUT/test_img/s2t/cvpdl_s2t/test_latest/images/fake_B
ls | head -2575 | xargs -i cp {} ../../../../../../../SSDA_dataset/source_data_fake/images/train
ls | tail -300 | xargs -i cp {} ../../../../../../../SSDA_dataset/source_data_fake/images/val
cd ../../../../../../../
cp ./yolotext/org/train/* ./SSDA_dataset/source_data_fake/labels/train
cp ./yolotext/org/val/* ./SSDA_dataset/source_data_fake/labels/val

cp ./yolotext/org/train/* ./SSDA_dataset/source_data/labels/train
cp ./yolotext/org/val/* ./SSDA_dataset/source_data/labels/val
cp $source_train_data/* ./SSDA_dataset/source_data/images/train
cp $source_val_data/* ./SSDA_dataset/source_data/images/val

mkdir -p SSDA_dataset/target_data/images/train
mkdir -p SSDA_dataset/target_data/images/train
mkdir -p SSDA_dataset/target_data/labels/val
mkdir -p SSDA_dataset/target_data/labels/val

mkdir -p SSDA_dataset/target_data_fake/images/train
cd ./CUT/test_img/t2s/cvpdl_t2s/test_latest/images/fake_B
ls | head -2575 | xargs -i cp {} ../../../../../../../SSDA_dataset/target_data_fake/images/train
cd ../../../../../../../
cp $target_train_data/* ./SSDA_dataset/target_data/images/train
cp $target_val_data/* ./SSDA_dataset/target_data/images/val
cp ./yolotext/fog/val/* ./SSDA_dataset/target_data/labels/val
cp ./yolot/runs/detect/exp/labels/* ./SSDA_dataset/target_data/labels/train


yq w -i data/yamls_sda/cvpdl2foggy.yaml train_source_real.[0] "dirname $(dirname $(dirname $(dirname "$source_train_data")))/SSDA_dataset/source_data/images/train"
yq w -i data/yamls_sda/cvpdl2foggy.yaml train_source_real.[1] "dirname $(dirname $(dirname $(dirname "$source_val_data")))/SSDA_dataset/source_data/images/val"
yq w -i data/yamls_sda/cvpdl2foggy.yaml train_source_fake.[0] "dirname $(dirname $(dirname $(dirname "$source_train_data")))/SSDA_dataset/source_data_fake/images/train"
yq w -i data/yamls_sda/cvpdl2foggy.yaml train_source_fake.[1] "dirname $(dirname $(dirname $(dirname "$source_val_data")))/SSDA_dataset/source_data_fake/images/val"
yq w -i data/yamls_sda/cvpdl2foggy.yaml train_target_real.[0] "dirname $(dirname $(dirname $(dirname "$target_train_data")))/SSDA_dataset/target_data/images/train"
yq w -i data/yamls_sda/cvpdl2foggy.yaml train_target_fake.[0] "dirname $(dirname $(dirname $(dirname "$target_train_data")))/SSDA_dataset/target_data_fake/images/train"
yq w -i data/yamls_sda/cvpdl2foggy.yaml test_target_real.[0] "dirname $(dirname $(dirname $(dirname "$target_val_data")))/SSDA_dataset/target_data/images/train"

cd ./SSDA-YOLO

python ssda_yolov5_train.py \
    --weights ../ssda_yolov5_5.0m.pt \
    --data ./data/yamls_sda/cvpdl2foggy.yaml \
    --name orgtofog \
    --img-size 640 \
    --device 0 \
    --batch-size 8 \
    --epochs 200 \
    --lambda_weight 0.005 \
    --consistency_loss \
    --alpha_weight 2.0 \
    --save_dir "$checkpoint_save_dir"
