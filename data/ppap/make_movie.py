
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
    bg=cv2.imread(filename,1)
    bg=cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
    bg=cv2.resize(bg,(300,200))
    bg=bg
    background_images.append(bg)

filename="ppap.avi"

fps=30
width=300
height=200

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(filename, int(fourcc), fps, (int(width), int(height)))

bg = background_images[0]
fgs = [foreground_images[1][0], foreground_images[0][0]]

# pp_pos=np.array([-100.0,  200.0]) #x,y
# pp_v = np.array([2.0,  -4.0]) #dx,dy

# pp_theta=0
# pp_omega=3.0
# pp_scale=1.0

# ap_pos=np.array([300.0, 100.0]) #x,y
# ap_v = np.array([-2.0,  -3.0]) #dx,dy

# ap_theta=0
# ap_omega=-3.0
# ap_scale=1.0


class object:
    def __init__(self,the_id,pos,v,theta, omega, scale):
        self.pos=np.array(pos,dtype=np.float)
        self.id=the_id
        self.v=np.array(v,dtype=np.float)
        self.theta=theta
        self.omega=omega
        self.scale=scale

    def step(self):
        self.pos +=self.v
        self.v[1] += 0.05 # G
        self.theta += self.omega


# objects=[
#     object(0, [-100,200], [2, -4], 0, 3.0, 1.0),
#     object(1, [ 200,200], [-2,-4], 0, -3.0, 1.0),
#     object(1, [-200,100], [5, -3], 90, 5.0, 0.5),
#     object(0, [ 600,100], [-5,-3], 90, -5.0, 0.5),
#     object(1, [-200,50], [10,  0], 90, 5.0, 1.2),
#     object(0, [ 600,50], [-10, 0], 90, -5.0, 1.2)
# ]

objects=[
    object(0, [-100,80], [5,-3], 0,   3.0,   1.0),
    object(1, [-200,200], [5,-3.2], 45, -3.0, 0.8),
    object(1, [-300,90], [5,-3.3], 90,  5.0, 1.3),
    object(0, [-400,210], [5,-3.4], 135, -5.0, 0.9),
    object(1, [-500,95], [5,-3.5], 180, 5.0, 1.2),
    object(0, [-600,200], [5,-3.6], 135, -5.0, 1.2),
    object(1, [-700,100], [5,-4], 180, 5.0, 1.0),
    object(0, [-800,105], [5,-5], 135, -5.0, 0.9)
]



frame_num=0

while True:

    frame = bg.copy()

    #print pp_pos, ap_pos

    for obj in objects:

        obj_r = rotate_image(fgs[obj.id].copy(), obj.theta, obj.scale, crop=False)
        frame = set_obj( frame, obj_r, obj.pos)
        frame = np.clip( (frame*255).astype(np.uint8),0,255)
        obj.step()


    #frame=(frame*255).astype(np.uint8)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    out.write(frame)

    if frame_num>5*fps:
        break
    frame_num+=1

