#!/bin/sh

python inference.py --data=../config/voc.data --pretrained_weights=../weights/yolov3.weights --model=../config/yolov3.cfg
