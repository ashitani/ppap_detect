import cv2
import numpy as np

import chainer
import chainer.functions as F

class Box():
  def __init__(self,x,y,w,h):
    self.x=x
    self.y=y
    self.w=w
    self.h=h

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1>l2 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1<r2 else r2
    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if(w < 0 or h < 0):
      return 0
    area = w*h
    return area

def box_union(a,b):
    i = box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);


def sigmoid(x):
  return 1.0/(np.exp(-x)+1.0)

def softmax(x):
  x=np.array([x]) #.reshape(1,len(x))
  return F.softmax(x).data

def draw_boxes(im_org,sorted_boxes,classes,block_x,block_y,biases,colors):
  im_marked=im_org.copy()
  im_size=np.shape(im_org)
  im_h=im_size[0]
  im_w=im_size[1]

  for sorted_box in sorted_boxes:
      b,j,class_id,p_class = sorted_box

      print classes[class_id], np.max(p_class)*100

      x=b.x
      y=b.y
      w=b.w
      h=b.h

      x0 = int(np.clip(x-w/2,0,im_w))
      y0 = int(np.clip(y-h/2,0,im_h))
      x1 = int(np.clip(x+w/2,0,im_w))
      y1 = int(np.clip(y+h/2,0,im_h))
      im_marked=cv2.rectangle(im_marked, (x0, y0),(x1, y1),colors[class_id],thickness=2)
#      im_marked=cv2.rectangle(im_marked, (x0, y0),(x0+100, y0+20) ,colors[class_id],thickness=-1)
#      cv2.putText(im_marked, classes[class_id],(x0+5,y0+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),thickness=2)
  return im_marked

def get_boxes(ans, block_x, block_y, bb_num,class_num,th, im_w,im_h,biases):
  sorted_boxes = []
  for by in range(block_y):
    for bx in range(block_x):
      for j in range(bb_num):

        box   = ans[by,bx,j,0:4]
        conf  = sigmoid(ans[by,bx,j,4])
        probs = softmax(ans[by,bx,j,5:(5+class_num)])[0]

        p_class = probs*conf

        if np.max(p_class)<th:
          continue
        class_id = np.argmax(p_class)

        x = (bx+sigmoid(box[0]))*(im_w/float(block_x))
        y = (by+sigmoid(box[1]))*(im_h/float(block_y))
        w = np.exp(box[2])*biases[j][0]*(im_w/float(block_x))
        h = np.exp(box[3])*biases[j][1]*(im_h/float(block_y))
        b = Box(x,y,w,h)

        sorted_boxes.append([b,j,class_id, max(p_class)])
  return sorted_boxes

def go_cnn(model, im_org,img_x,img_y,block_x,block_y,bb_num,class_num):
  im0=cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)
  im0=cv2.resize(im0,(img_y,img_x))
  im=np.asarray(im0,dtype=np.float32)/255.0

  ans = model.predict( im.transpose(2,0,1).reshape(1,3,img_y,img_x)).data[0]
  ans = ans.transpose(1,2,0) # (13,13,125)
  ans = ans.reshape(block_y,block_x,bb_num,(bb_num+class_num))
  return ans

def sort_boxes(boxes, class_num, nms):
  import itertools
  import copy
  from operator import itemgetter

  sorted_boxes = sorted(boxes, key=itemgetter(2,3), reverse=True)
  buf = copy.copy(sorted_boxes)

  total=len(sorted_boxes)
  # for i,a_sb in enumerate(sorted_boxes):
  #   a=a_sb
  #   if a[3]<0.25:
  #     continue
  #   for j in range(i+1,total):
  #     b=sorted_boxes[j]
  #     if a[2]==b[2] and box_iou(a[0],b[0])>0.5:
  #       if b in buf:
  #         buf.remove(b)

  for a_sb,b_sb  in itertools.combinations(sorted_boxes,2):
    a=a_sb
    b=b_sb
    if a[2]==b[2] and box_iou(a[0],b[0])>nms and a[3]>b[3]:
      if b in buf:
        buf.remove(b)
  return buf

def predict(model,im_org):

  # from tiny-yolo-voc.cfg
  img_x=416
  img_y=416
  block_x=13
  block_y=13
  class_num=2
  bb_num=5
  biases =[[ 1.08,1.19],[3.42,4.41],[6.63,11.38],[9.42,5.11],[16.62,10.52]]
  # th = 0.85
  # nms = 0.5
  th = 0.8
  nms = 0.05

  # classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
  #            "bus", "car", "cat", "chair", "cow",
  #            "diningtable", "dog", "horse", "motorbike", "person",
  #            "pottedplant", "sheep", "sofa", "train","tvmonitor"]
  classes = ["PP", "AP"]
  colors=[(0,255,255),(0,0,255)]

  im_size=np.shape(im_org)
  im_h=im_size[0]
  im_w=im_size[1]

  # get CNN output
  ans = go_cnn(model,im_org, img_x,img_y,block_x,block_y,bb_num,class_num)

  # get bounding boxes
  boxes = get_boxes(ans, block_x, block_y, bb_num, class_num, th, im_w,im_h, biases)

  # Sort and remove intersection
  sorted_boxes=sort_boxes(boxes, class_num,nms)

  # Draw boxes
  im_marked = draw_boxes(im_org,sorted_boxes,classes, block_x,block_y,biases,colors)

  return im_marked