import json
import argparse
import os


import argparse

msg = "Adding argment"
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="", help="target image path")
parser.add_argument('--output_dir', type=str, default="", help="your output json file") 

args = parser.parse_args()

input_dir = args.input_dir
output = args.output_dir

# 設定輸入和輸出檔案路徑
# input_path = "./hw3_dataset/fog/val.coco.json"
# output_path = "./dataset_yolo/fog/val"

# 如果輸出路徑不存在，就建立資料夾
if not os.path.exists(output):
    os.makedirs(output)

# 定義類別名稱對應的編號，需要與json檔案中的id對應
class_id_map = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "rider": 4,
    "motorcycle": 5,
    "bicycle": 6,
    "train": 7,
}
print(input_dir)
# 讀取 JSON 檔案
with open(input_dir, 'r') as f:
    data = json.load(f)

# 逐一處理每張圖片
for image in data["images"]:
    # 取得圖片名稱
    # image_name = image["file_name"].replace("org/train/", "")
    image_name = "/".join(image["file_name"].split("/")[-1:]) + ".jpg"

    # 取得圖片的寬度和高度
    image_width = image["width"]
    image_height = image["height"]

    # 打開對應的txt檔案，如果檔案不存在則建立檔案
    output_file = open(os.path.join(output, os.path.splitext(image_name)[0] + ".txt"), "w+")

    # 尋找所有跟這張圖片有關聯的annotation
    for annotation in data["annotations"]:
        if annotation["image_id"] == image["id"]:
            # 取得這個annotation對應的類別編號和bbox
            class_id = annotation["category_id"]-1
            bbox = annotation["bbox"]

            # 計算bbox的中心點座標和寬度、高度
            bbox_x = bbox[0] + bbox[2] / 2
            bbox_y = bbox[1] + bbox[3] / 2
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            # 將bbox座標、寬度和高度轉換為yolo格式
            yolo_x = bbox_x / image_width
            yolo_y = bbox_y / image_height
            yolo_w = bbox_w / image_width
            yolo_h = bbox_h / image_height


            # 將資訊寫入txt檔案
            output_file.write(f"{class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

    # 關閉txt檔案
    output_file.close()
