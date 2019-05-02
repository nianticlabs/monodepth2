# Our standard mono+stereo model
python ../train.py --model_name MS_640x192 \
  --use_stereo --frame_ids 0 -1 1

# Our low resolution mono+stereo model
python ../train.py --model_name MS_416x128 \
  --use_stereo --frame_ids 0 -1 1 \
  --height 128 --width 416

# Our high resolution mono+stereo model
python ../train.py --model_name MS_1024x320 \
  --use_stereo --frame_ids 0 -1 1 \
  --height 320 --width 1024 \
  --load_weights_folder ~/tmp/MS_640x192/models/weights_9 \
  --num_epochs 5 --learning_rate 1e-5

# Our standard mono+stereo model w/o pretraining
python ../train.py --model_name MS_640x192_no_pt \
  --use_stereo --frame_ids 0 -1 1 \
  --weights_init scratch \
  --num_epochs 30

# Baseline mono+stereo model, i.e. ours with our contributions turned off
python ../train.py --model_name MS_640x192_baseline \
  --use_stereo --frame_ids 0 -1 1 \
  --v1_multiscale --disable_automasking --avg_reprojection

# Mono+stereo without full-res multiscale
python ../train.py --model_name MS_640x192_no_full_res_ms \
  --use_stereo --frame_ids 0 -1 1 \
  --v1_multiscale

# Mono+stereo without automasking
python ../train.py --model_name MS_640x192_no_automasking \
  --use_stereo --frame_ids 0 -1 1 \
  --disable_automasking

# Mono+stereo without min reproj
python ../train.py --model_name MS_640x192_no_min_reproj \
  --use_stereo --frame_ids 0 -1 1 \
  --avg_reprojection
