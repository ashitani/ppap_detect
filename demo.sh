#!/bin/sh
./darknet detector test cfg/ppap.dataset cfg/tiny-yolo-ppap.cfg /tmp/ppap-backup/tiny-yolo-ppap_final.weights data/ppap/images2/00005.png -thresh 0.9
