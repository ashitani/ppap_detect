#!/usr/bin/env python

from YOLOtiny_chainer_v2 import *
import cv2
import sys
import os

if len(sys.argv)!=2:
     print "Usage: python replay_movie.py filename"
     exit(-1)
filename=sys.argv[1]
cap = cv2.VideoCapture(filename)

print "Loading model"
model=YOLOtiny()
serializers.load_npz('YOLOtiny_chainer_v2/YOLOtiny_v2.model',model)


outfile="out.avi"
fps=30
width=296
height=200
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(outfile, int(fourcc), fps, (int(width), int(height)))

frame_num=0
while(True):
    print frame_num
    ret, im_org = cap.read()
    if ret==False:
        break
    im_org=im_org[:,0:296,:]

    # if frame_num>30:
    #     break
    im_marked = predict(model,im_org)

    path, ext = os.path.splitext(os.path.basename(filename))

    #im_marked=cv2.cvtColor(im_marked,cv2.COLOR_BGR2RGB)
    out.write(im_marked)
    frame_num+=1

