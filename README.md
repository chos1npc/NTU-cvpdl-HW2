# Domain Adaptation object detection
Course in NTU:computer vision practice with deep learning

## SSDA-YOLO: Semi-supervised Domain Adaptive YOLO for Cross-Domain Object Detection

## Introduction
This project implements SSDA-YOLO on our custom dataset, a domain adaptation method for object detection. The approach is based on semi-supervised learning and transfer learning techniques, enabling YOLOv5 to perform well across different domains. The method includes a teacher-student framework, pseudo-labeling, and a consistency loss to improve domain adaptation performance.

## Repository Structure
```
├── yolov5/                # YOLOv5 object detector
├── SSDA-YOLO/            # Semi-Supervised Domain Adaptation YOLO
├── CUT/                  # Contrastive Unsupervised Transfer Learning
├── scripts/
│   ├── hw3_download.sh   # Script to download model checkpoints
│   ├── hw3_inference.sh  # Script to run inference
│   ├── train.sh          # Training script
│   ├── jsontrans.py      # YOLO label format converter
│   ├── txtrans.py        # Converts dataset to the required format
├── datasets/             # Dataset directory
├── results/              # Output results
├── README.md             # This documentation
```

## Installation
Before running the project, install dependencies:
```bash
# Install YOLOv5 dependencies
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..

# Install SSDA-YOLO dependencies
git clone <SSDA-YOLO-REPO>
cd SSDA-YOLO
pip install -r requirements.txt
cd ..

# Install CUT dependencies
git clone <CUT-REPO>
cd CUT
pip install -r requirements.txt
cd ..
```

## Training Process
### Step 1: Convert Labels
Convert dataset labels to the required YOLOv5 and SSDA-YOLO formats:
```bash
python txtrans.py --input_dir path/to/source/train.coco.json --output_dir ./yolotext/org/train
python txtrans.py --input_dir path/to/source/val.coco.json --output_dir ./yolotext/org/val
python txtrans.py --input_dir path/to/target/val.coco.json --output_dir ./yolotext/fog/val
```

### Step 2: Generate Source-Fake and Target-Fake Images
Generate domain-adapted images using CUT:
```bash
bash train.sh /path/to/source/train /path/to/source/val /path/to/target/train /path/to/target/val /path/to/save/checkpoints
```

### Step 3: Train the Model
Modify the dataset path in the YAML file, then start training SSDA-YOLO:
```bash
python train.py --config ssda_yolo.yaml --epochs 100 --batch-size 16
```

## Inference
Run inference using a trained model:
```bash
bash hw3_inference.sh /path/to/image /path/to/output 0
```

## Model Checkpoints
You can download trained model checkpoints using:
```bash
bash hw3_download.sh
```

## Citation
```
@article{ssda-yolo,
  title={Source-Free Semi-Supervised Domain Adaptation YOLO},
  author={Your Name},
  journal={Arxiv},
  year={2025}
}
```

