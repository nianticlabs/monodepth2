MODEL_NAME='finetuned'

nohup ./../train.py --model_name $MODEL_NAME --split nyu --dataset nyu --data_path /mnt/disks/data/nyudepth/ --max_depth=10 --batch_size 6 --num_epochs 15  --height 256 --width 352 --load_weights_folder ~/tmp/base_model/models/weights_19 &
