MODEL_NAME='mono_base_model_full2'

nohup ./../train.py --model_name $MODEL_NAME --data_path  /mnt/disks/data/kitti/ --batch_size 4 &
