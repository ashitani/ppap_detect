
import cv2
import numpy as np
import glob

import os

from common import *

import random

foreground_images = []
background_images = []

for obj_id, folder in enumerate(glob.glob("foreground/*")):
    fgs=[]
    for filename in glob.glob(folder+"/*" ):
        fg=cv2.imread(filename,-1)
        fg=cv2.cvtColor(fg,cv2.COLOR_BGRA2RGBA)
        fg=fg
        fgs.append(fg)
    foreground_images.append(fgs)


train_data_num = 100
valid_data_num =  20

classes = 2

angle_range = [-180,180]
scale_range = [0.5,2.0]

image_folder ="images_pre"
train_txt ="ppap_train_pre.txt"
valid_txt ="ppap_valid_pre.txt"

fw_train = open(train_txt,"w")
fw_valid = open(valid_txt,"w")

test_data_num = train_data_num+valid_data_num


for img_id in range(test_data_num):

    class_id=random.randint(0,1)

    if img_id<train_data_num:
        image_filename = image_folder+"/t_%05d_c%01d.png"%(img_id,class_id)
    else:
        image_filename = image_folder+"/v_%05d_c%01d.png"%(img_id,class_id)


    angle = random.uniform(0,180)
    dh = random.randint(0,10)
    dw = random.randint(0,36)

    fg=foreground_images[class_id][0]
    img=rotate_image(fg,angle,1.0)

    img=cv2.copyMakeBorder(img,dh,dh,dw,dw,cv2.BORDER_CONSTANT,value=(0,0,0,0))

    img=cv2.resize(img, (64,64))
    img=cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(image_filename, img)

    if img_id<train_data_num:
        fw_train.write("%s\n" % os.path.abspath(image_filename))
    else:
        fw_valid.write("%s\n" % os.path.abspath(image_filename))

fw_train.close()
fw_valid.close()

