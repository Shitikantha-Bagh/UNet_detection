import json
import numpy as np
import cv2
import os

def label_json(json_path):
    with open(json_path,'r') as f:
        data=json.load(f)
    return data

def extract_points(label_data):
    annotations=[]
    for shape in label_data['shapes']:
        label=shape['label']
        if label in ["ShapeL","ShapeVertical","ShapeRound","ShapeLine"]:
            points= shape['points']
            annotations.append((label,points))
    return annotations

def create_binary_mask(image_shape,annotations):
    mask= np.zeros(image_shape[:2],dtype=np.uint8)
    for label, points in annotations:
        points = np.array(points,dtype=np.int32)
        cv2.fillPoly(mask,pts=[points],color=(255,255,255))
    return mask

path = "/home/sbagh/Documents/UNet_Detection/datasets/narrowsets_092623/"


for folder_name in os.listdir(path):
    folder_path = os.path.join(path,folder_name)
    
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_path = os.path.join(folder_path,filename)
                label_data = label_json(json_path)
                annotations = extract_points(label_data)

                image_height=label_data["imageHeight"]
                image_width=label_data["imageWidth"]

                mask = create_binary_mask((image_height,image_width),annotations)

                mask_filename = filename.replace('_imgSets_','_joint_labelIds_').replace('.json','.png')
                mask_path = os.path.join(folder_path,mask_filename)
                cv2.imwrite(mask_path,mask)
