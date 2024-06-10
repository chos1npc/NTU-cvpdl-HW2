# training dataset access
希望輸入的dataset格式跟作業給的dataset一致
輸入參數共有五個
source_train_data=$1    #/home/mmlab206/CVPDL/HW3/hw3_dataset/org/train
source_val_data=$2  #/home/mmlab206/CVPDL/HW3/hw3_dataset/org/val
target_train_data=$3    #/home/mmlab206/CVPDL/HW3/hw3_dataset/fog/train
target_val_data=$4  #/home/mmlab206/CVPDL/HW3/hw3_dataset/fog/val
checkpoint_save_dir=$5

要是完整的path
org label放在HW3/hw3_dataset/org底下，有train.coco.json和val.coco.json兩個檔案
fog label放在HW3/hw3_dataset/fog底下，有train.coco.json和val.coco.json兩個檔案
train.sh會做讀取，並依照README.md給的process做training
'''
第一步:先將label轉為yolov5以及ssda-yolo所需要的格式
第二步:利用cut生成source fake image和target fake image 需要進行兩次的訓練 分別從source->target和target->source
第三步:source fake的lable與原本的source一樣，target的label是我們利用yolov5 detect source得到的pseudo label，target_fake不需要標籤
第四步:將所有的dir建成一個dataset 有source,source_fake,target,target_fake四個資料夾，source,source_fake,target內皆有image,label和train,val，target_fake僅有image和train
第五步:更改yaml file，將dataset path做修改
第六步:開始train ssda-yolo
'''
可能會有bug的地方在第五步，因不確定助教端電腦的path