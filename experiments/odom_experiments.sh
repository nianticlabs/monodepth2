# A different kitti dataset is required for odometry training and evaluation.
# This can be downloaded from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
# We assume this has been extraced to the folder ../kitti_data_odom

# Standard mono odometry model.
python ../train.py --model_name M_odom \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom

# Mono odometry model without Imagenet pretraining
python ../train.py --model_name M_odom_no_pt \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --weights_init scratch --num_epochs 30

# Mono + stereo odometry model
python ../train.py --model_name MS_odom \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --use_stereo

# Mono + stereo odometry model without Imagenet pretraining
python ../train.py --model_name MS_odom_no_pt \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --use_stereo \
  --weights_init scratch --num_epochs 30
