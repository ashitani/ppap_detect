
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

for filename in glob.glob("background/*"):
    print filename
    bg=cv2.imread(filename,1)
    bg=cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
    bg=cv2.resize(bg,(300,200))
    bg=bg
    background_images.append(bg)

print np.shape(foreground_images)
print np.shape(background_images)

train_data_num = 1000
valid_data_num = 32

objects_per_img = 1
classes = 2

#bs = [600,400] # size of background_image (x,y)

angle_range = [-180,180]
scale_range = [0.5,2.0]

image_folder ="images"
label_folder ="labels"
train_txt ="ppap_train.txt"
valid_txt ="ppap_valid.txt"

fw_train = open(train_txt,"w")
fw_valid = open(valid_txt,"w")

test_data_num = train_data_num+valid_data_num

for img_id in range(test_data_num):

    image_filename = image_folder+"/%0005d.png" % img_id
    label_filename = label_folder+"/%0005d.txt" % img_id

    bg_id=random.randint(0,len(background_images)-1)

    bg = background_images[bg_id].copy()

    print img_id ,"/", test_data_num
    fw=open(label_filename,"w")

    for obj in range(objects_per_img):
        while(1):
            class_id = random.randint(0,classes-1)
            angle = random.uniform(angle_range[0],angle_range[1])
            scale  = random.uniform(scale_range[0], scale_range[1])
            bs=np.shape(bg)
            x = random.randint(0, int(bs[1]*0.8))
            y = random.randint(0, int(bs[0]*0.8))

            obj_id = random.randint(0,len(foreground_images[class_id])-1)
            fg = foreground_images[class_id][obj_id].copy()
            bg, bb= aug_img(bg,fg, angle, scale,[x,y])
            if bb!=None:
                bg = (bg*255).astype(np.uint8)

            if bb==None:
                pass
                #print "out_of_range"
            else:
                x,y,w,h = bb
                cv2.rectangle(fg, (x,y),(x+w,y+h), color=(1.0,0,0), thickness=4)

                bb_x = (x+w/2.0 )/bs[1]
                bb_y = (y+h/2.0 )/bs[0]
                bb_w = float(w) /bs[1]
                bb_h = float(h) /bs[0]
                fw.write( "%d %f %f %f %f\n" % (class_id, bb_x, bb_y, bb_w, bb_h))
                #img[:,:,0:3]*=255
                #img=int(img)
                #img =np.asarray(img,dtype=np.uint8)
                break

    fw.close()
    outimg =(np.clip( bg[:,:,:3],0,255)).astype(np.uint8)
    outimg=cv2.cvtColor(outimg,cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_filename,outimg)
    if img_id<train_data_num:
        fw_train.write("%s\n" % os.path.abspath(image_filename))
    else:
        fw_valid.write("%s\n" % os.path.abspath(image_filename))

fw_train.close()
fw_valid.close()
