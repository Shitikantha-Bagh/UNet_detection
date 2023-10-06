import json
import numpy as np
import shutil
import os
import random

path = "/home/sbagh/Documents/UNet_Detection/datasets/narrowsets_092623/"
directories= ['train','val','test']


for directory in directories:
    os.makedirs(os.path.join(path,f'{directory}_images',directory),exist_ok=True)
    os.makedirs(os.path.join(path,f'{directory}_masks',directory),exist_ok=True)

                     
folder_name =['shapeRound','shapeL','shapeLine','shapeVertical']


all_images=[]
all_masks=[]

for folder_name in os.listdir(path):
    folder_path = os.path.join(path,folder_name)
    
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith("_imgSets_1024x512.png"):
                image_path = os.path.join(folder_path,filename)
                mask_filename = filename.replace("_imgSets_","_joint_labelIds_")
                mask_path = os.path.join(folder_path,mask_filename)
                if os.path.isfile(mask_path):
                    all_images.append(image_path)
                    all_masks.append(mask_path)

# print(all_images[1])
# print(all_masks[1])
combined = list(zip(all_images,all_masks))
random.shuffle(combined)
all_images,all_masks = zip(*combined)

total_samples = len(all_images)
train_split = int(0.7* total_samples)
val_split = int(0.2*total_samples)

train_images = all_images[:train_split]
val_images = all_images[train_split:train_split+val_split]
test_images= all_images[train_split+val_split:]

train_masks = all_masks[:train_split]
val_masks = all_masks[train_split:train_split+val_split]
test_masks= all_masks[train_split+val_split:]

image_subdirectory = 'images'
mask_subdirectory='masks'

def copy_files(src_files,dst_directory,subdirectory,subsub_directory,prefix):
    for i, src_path in enumerate (src_files,start=1):
        filename = f'{prefix}{i}.png'
        dst_path = os.path.join(dst_directory,subdirectory,subsub_directory,filename)
        shutil.copyfile(src_path,dst_path)

copy_files(train_images,path,'train_images','train','image')
copy_files(train_masks,path,'train_masks','train','mask')
copy_files(val_images,path,'val_images','val','image')
copy_files(val_masks,path,'val_masks','val','mask')
copy_files(test_images,path,'test_images','test','image')
copy_files(test_masks,path,'test_masks','test','mask')
