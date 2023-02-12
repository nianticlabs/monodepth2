#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=14 ; var < 20 ; var++));
do
  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_original/models/weights_$var --eval_mono
done