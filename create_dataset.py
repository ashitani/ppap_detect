
import cv2
import numpy as np
import glob

import os

def set_obj(bg,fg, pos):
    ret=bg.copy()
    mask = fg[:,:,3]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask/ 255.0
    fg_rgb=fg[:,:,:3]
    fg_rgb=fg_rgb/255.0
    ret=ret/255.0

    x,y = pos
    siz=np.shape(fg_rgb)

    ret[y:(y+siz[0]),x:(x+siz[1]),:]*= (1-mask)
    ret[y:(y+siz[0]),x:(x+siz[1]),:]+= fg_rgb
    #ret*=255
    #ret=np.clip(ret,0,255)
    #ret=ret.astype(np.uint8)
    return ret


def pad_image(img_src,scale):
    size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))
    org_h=size[1]
    org_w=size[0]

    src_r = np.sqrt((size[0]/2.0)**2+(size[1]/2.0)**2)
    dest_h = int(2*src_r * scale)
    dest_w = int(2*src_r * scale)

    dh= (dest_h-org_h)/2
    dw= (dest_w-org_w)/2

    img=img_src
    if dh>0:
        img=cv2.copyMakeBorder(img,dh,dh,0,0,cv2.BORDER_CONSTANT,value=(0,0,0,0))
    if dw>0:
        img=cv2.copyMakeBorder(img,0,0,dw,dw,cv2.BORDER_CONSTANT,value=(0,0,0,0))
    return img, [dest_h,dest_w]


def rotate_image(img_src, angle,scale ):
    img_src,size_dest= pad_image(img_src,scale)

    size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))
    org_h=size[1]
    org_w=size[0]

    src_r = np.sqrt((size[0]/2.0)**2+(size[1]/2.0)**2)
    org_angle =np.arctan(float(org_h)/org_w)

    dest_h = size_dest[0]
    dest_w = size_dest[1]

    center = tuple(np.array([img_src.shape[1] * 0.5, img_src.shape[0] * 0.5]))

    dsize= (dest_w,dest_h)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img_rot = cv2.warpAffine(img_src, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    x,y,w,h = cv2.boundingRect(img_rot[:,:,3])
    return img_rot[y:y+h, x:x+w,:]

def aug_img( bg, fg, angle=45, scale=1.0, pos=[100,100]):
    r_img = rotate_image(fg.copy(), angle,scale)
    r_siz = np.shape(r_img)

    bg_siz = np.shape(bg)

    x=pos[0]
    y=pos[1]
    w=r_siz[1]
    h=r_siz[0]
    border_pos = [x,y,w,h]

    if x+w>bg_siz[1] or y+h>bg_siz[0] :
        return bg, None
    ans = set_obj(bg,r_img,pos)
    return ans, border_pos

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
    bg=cv2.imread(filename,1)
    bg=cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
    bg=cv2.resize(bg,(600,400))
    bg=bg
    background_images.append(bg)


train_data_num = 100
valid_data_num =  20

objects_per_img = 8
classes = 2

bs = [400,600] # size of background_image (x,y)

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

    image_filename = "images/%0005d.png" % img_id
    label_filename = "labels/%0005d.txt" % img_id

    bg_id=random.randint(0,len(background_images)-1)

    bg = background_images[bg_id].copy()

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
                print "out_of_range"
            else:
                x,y,w,h = bb
                cv2.rectangle(fg, (x,y),(x+w,y+h), color=(1.0,0,0), thickness=4)

                bb_x = (x+w/2.0 )/bs[0]
                bb_y = (y+h/2.0 )/bs[1]
                bb_w = float(w) /bs[0]
                bb_h = float(h) /bs[1]
                fw.write( "%d, %f, %f, %f, %f\n" % (class_id, bb_x, bb_y, bb_w, bb_h))
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
