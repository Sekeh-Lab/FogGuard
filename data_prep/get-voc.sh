#!/usr/bin/env bash

DST="$HOME/data/VOC/"

cd $DST


wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar &
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar &
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar &

wait

tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py

cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt
