MODEL_NAME='nyu_trained'

nohup ./../train.py --model_name $MODEL_NAME --split nyu --dataset nyu --data_path  /mnt/disks/data/nyudepth/ --batch_size 6  --height 256 --width 352 --max_depth 10 &
