import json
import os
from PIL import Image

def parse_yolo_txt(img_path,txt_file,txtname):
    txtname = txtname.replace('.txt','.jpg')
    print(img_path)
    if txtname in os.listdir(img_path):
        print(txtname)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            lines = [i.strip('\n').split() for i in lines]
            img_name = os.path.splitext(os.path.basename(txt_file))[0] + '.jpg'
            print(img_name)
            img = Image.open(os.path.join(img_path,img_name))
            # get image size
            img_w, img_h = img.size

            boxes = []
            labels = []
            scores = []
            for line in lines[1:]:
                print(line)
                cls_id, x, y, w, h, score = line
                x =float(x)
                y =float(y)
                w =float(w)
                h =float(h)
                score = float(score)
                cls_id = int(cls_id)+1
                x1 = int((x - w / 2) * img_w)
                y1 = int((y - h / 2) * img_h)
                x2 = int((x + w / 2) * img_w)
                y2 = int((y + h / 2) * img_h)
                print(x1,y1,x2,y2)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))
                scores.append(score)
            print(scores)
            return {
                img_name: {
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores
                }
            }

def yolo_txt_to_json(img_path,txt_dir, json_file):
    data = {}
    for txt_name in os.listdir(txt_dir):
        if txt_name.endswith('.txt'):
            txt_path = os.path.join(txt_dir, txt_name)
            img_data = parse_yolo_txt(img_path,txt_path,txt_name)
            # print(img_data)
            data.update(img_data)

    with open(json_file, 'w') as f:
        json.dump(data, f)


import argparse

msg = "Adding argment"
parser = argparse.ArgumentParser()
parser.add_argument('--testimg_dir', type=str, default="", help="target image path")
parser.add_argument('--output_dir', type=str, default="", help="your output json file") 
parser.add_argument('--yolo_txt', type=str, default="", help="yolo detect output file path")

args = parser.parse_args()


yolo_txt = args.yolo_txt 
test_img_path = args.testimg_dir
output = args.output_dir

yolo_txt_to_json(img_path = test_img_path,txt_dir=yolo_txt,json_file=output)
