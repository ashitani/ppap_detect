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
    x=int(x)
    y=int(y)

    siz=np.shape(fg_rgb)

    bg_siz=np.shape(bg)

    # clip target position
    ymin=np.clip(y,0,bg_siz[0])
    xmin=np.clip(x,1,bg_siz[1])

    ymax = np.clip(y+siz[0],0,bg_siz[0])
    xmax = np.clip(x+siz[1],0,bg_siz[1])

    # clip source position
    if y<0:
        ymin_t = ymin-y
        ymax_t = siz[0]
    else:
        ymin_t = 0
        ymax_t = ymax-ymin
    if x<0:
        xmin_t = xmin-x
        xmax_t = siz[1]
    else:
        xmin_t = 0
        xmax_t = xmax-xmin

    #print y,ymin,ymax, ymin_t, ymax_t

    ret[ymin:ymax, xmin:xmax, :]*= (1-mask[ymin_t:ymax_t,xmin_t:xmax_t,:])
    ret[ymin:ymax, xmin:xmax, :]+= fg_rgb[ymin_t:ymax_t,xmin_t:xmax_t,:]
    return ret


def set_obj_(bg,fg, pos):
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


def rotate_image(img_src, angle,scale ,crop=True):
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

    if crop:
        x,y,w,h = cv2.boundingRect(img_rot[:,:,3])
        return img_rot[y:y+h, x:x+w,:]
    else:
        return img_rot

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