# YOLO_PPAP

PP/AP identification using YOLO_ver2 and chainer.
For detail, see [Qiita Entry](http://qiita.com/ashitani/items/566cf9234682cb5f2d60).

![example_movie](https://raw.githubusercontent.com/ashitani/ppap_detect/master/doc/out01.gif)


# Preparation

```
git clone this_repositry
ln -s /path/to/darknet .
ln -s /path/to/darknet/data/labels ./data/labels
```

# Create materials

- Create your own PP/AP (PNG files with transparent background)
- Place them to data/ppap/foreground/00/ and data/ppap/foreground/01/
- Create your own background images
- Place them to data/ppap/background

# Create dataset

```
cd data/ppap
mkdir images_pre
mkdir images
mkdir labels
python create_pretrain_dataset.py
python create_dataset.py
cd ../..
```

# Pre-train & conert pre-train weights to initial weight

```
mkdir /tmp/backup
./pretrain.sh
./convert.sh
```

# Train

```
mkdir /tmp/ppap-backup
./train.sh
cp /tmp/ppap-backup/tiny-yolo-final.weights ./YOLOtiny_chainer_v2
```

# Convert darknet weights to chainer

```
cd ./YOLOtiny_chainer_v2
python YOLOtiny.py
cd ..
```

# Prediction

## predict image file

```
python replay_file.py filename.png
```
Output is written to filename_out.png

## predict image files in data/ppap/images/


```
mkdir outfiles
python replay.py
```
Output is written to outfiles foldes

## predict avi file

```
python replay_movie.py filename
```
Output is written to out.avi

