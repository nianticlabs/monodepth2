MODEL_NAME='finetuned_more'

nohup ./../train.py --model_name $MODEL_NAME --split nyu --dataset nyu --data_path /mnt/disks/data/nyudepth/ --max_depth=10 --batch_size 6 --num_epochs 20 --learning_rate 0.001  --height 256 --width 352 --load_weights_folder ~/tmp/base_model/models/weights_19  --frame_ids 0 -1 1 -2 2 3 -3 &
