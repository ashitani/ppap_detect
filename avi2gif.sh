#!/bin.sh
ffmpeg -i out00.avi  -an -r 30 -s 210x140  -pix_fmt rgb8 -f gif out00.gif
ffmpeg -i out01.avi  -an -r 30 -s 210x140  -pix_fmt rgb8 -f gif out01.gif
