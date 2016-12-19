#!/usr/bin/env python

from YOLOtiny_chainer_v2 import *
import cv2
import sys
import os

# if len(sys.argv)!=2:
#     print "Usage: python replay.py filename"
#     exit(-1)
# filename=sys.argv[1]

print "Loading model"
model=YOLOtiny()
serializers.load_npz('YOLOtiny_chainer_v2/YOLOtiny_v2.model',model)

for i in range(160):

    filename="data/ppap/images/%05d.png" % i

    im_org=cv2.imread(filename)
    im_marked = predict(model,im_org)

    path, ext = os.path.splitext(os.path.basename(filename))
    outfile="outfiles/%05d.png" % i
    cv2.imwrite(outfile,im_marked)
    print "Result was written to ",outfile
