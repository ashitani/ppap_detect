#!/bin/sh
./darknet classifier predict cfg/ppap-pre.dataset cfg/tiny-yolo-pretrain.cfg ./tiny-yolo-pretrain.weights ./data/ppap/images_pre/t_00001_c0.png
./darknet classifier predict cfg/ppap-pre.dataset cfg/tiny-yolo-pretrain.cfg ./tiny-yolo-pretrain.weights ./data/ppap/images_pre/t_00001_c1.png
