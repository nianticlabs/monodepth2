# Monodepth2

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [ICCV 2019](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```



## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
We ran our experiments with PyTorch 0.4.1, CUDA 9.1, Python 3.6.6 and Ubuntu 18.04.
We have also successfully trained models with PyTorch 1.0, and our code is compatible with Python 2.7. You may have issues installing OpenCV version 3.3.1 if you use Python 3.7, we recommend to create a virtual environment with Python 3.6.6 `conda create -n monodepth2 python=3.6.6 anaconda `.

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->


## 🖼️ Prediction for a single image

You can predict depth for a single image with:
```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

On its first run this will download the `mono+stereo_640x192` pretrained model (99MB) into the `models/` folder.
We provide the following  options for `--model_name`:


| `--model_name`          | Training modality | Imagenet pretrained? | Model resolution  | KITTI abs. rel. error |  delta < 1.25  |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|
| [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip)          | Mono              | Yes | 640 x 192                | 0.115                 | 0.877          |
| [`stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip)        | Stereo            | Yes | 640 x 192                | 0.109                 | 0.864          |
| [`mono+stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip)   | Mono + Stereo     | Yes | 640 x 192                | 0.106                 | 0.874          |
| [`mono_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip)         | Mono              | Yes | 1024 x 320               | 0.115                 | 0.879          |
| [`stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip)       | Stereo            | Yes | 1024 x 320               | 0.107                 | 0.874          |
| [`mono+stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip)  | Mono + Stereo     | Yes | 1024 x 320               | 0.106                 | 0.876          |
| [`mono_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip)          | Mono              | No | 640 x 192                | 0.132                 | 0.845          |
| [`stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip)        | Stereo            | No | 640 x 192                | 0.130                 | 0.831          |
| [`mono+stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip)   | Mono + Stereo     | No | 640 x 192                | 0.127                 | 0.836          |

You can also download models trained on the odometry split with [monocular](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_odom_640x192.zip) and [mono+stereo](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_odom_640x192.zip) training modalities.
Finally, we provide resnet 50 depth estimation models trained with [ImageNet pretrained weights](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_640x192.zip) and [trained from scratch](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_no_pt_640x192.zip).

## 💾 KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

The above conversion command creates images which match our experiments, where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 with default chroma subsampling `2x2,1x1,1x1`.
We found that Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which gives different results, hence the explicit parameter in the conversion command.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model
```

**Stereo training:**

Our code defaults to using Zhou's subsampled Eigen training data. For stereo-only training we have to specify that we want to use the full Eigen training set – see paper for details.
```shell
python train.py --model_name stereo_model \
  --frame_ids 0 --use_stereo --split eigen_full
```

**Monocular + stereo training:**
```shell
python train.py --model_name mono+stereo_model \
  --frame_ids 0 -1 1 --use_stereo
```


### GPUs

The code can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

All our experiments were performed on a single NVIDIA Titan Xp.

| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |
| Stereo            | 6GB                     | 8 hours                     |
| Mono + Stereo     | 11GB                    | 15 hours                    |



### 💽 Finetuning a pretrained model

Add the following to the training command to load an existing model for finetuning:
```shell
python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19
```


### 🔧 Other training options

Run `python train.py -h` (or look at `options.py`) to see the range of other training options, such as learning rates and ablation settings.


## 📊 KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono
```
For stereo models, you must use the `--eval_stereo` flag (see note below):
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_stereo
```
If you train your own model with our code you are likely to see slight differences to the publication results due to randomization in the weights initialization and data loading.

An additional parameter `--eval_split` can be set.
The three different values possible for `eval_split` are explained here:

| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (default) or `--split eigen_full` | The standard Eigen test files |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (default) or `--split eigen_full`  | Evaluate with the improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| **`benchmark`**       | 500           | `--split benchmark`        | The [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files. |

Because no ground truth is available for the new KITTI depth benchmark, no scores will be reported  when `--eval_split benchmark` is set.
Instead, a set of `.png` images will be saved to disk ready for upload to the evaluation server.


**External disparities evaluation**

Finally you can also use `evaluate_depth.py` to evaluate raw disparities (or inverse depth) from other methods by using the `--ext_disp_to_eval` flag:

```shell
python evaluate_depth.py --ext_disp_to_eval ~/other_method_disp.npy
```


**📷📷 Note on stereo evaluation**

Our stereo models are trained with an effective baseline of `0.1` units, while the actual KITTI stereo rig has a baseline of `0.54m`. This means a scaling of `5.4` must be applied for evaluation.
In addition, for models trained with stereo supervision we disable median scaling.
Setting the `--eval_stereo` flag when evaluating will automatically disable median scaling and scale predicted depths by `5.4`.


**⤴️⤵️ Odometry evaluation**

We include code for evaluating poses predicted by models trained with `--split odom --dataset kitti_odom --data_path /path/to/kitti/odometry/dataset`.

For this evaluation, the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) **(color, 65GB)** and **ground truth poses** zip files must be downloaded.
As above, we assume that the pngs have been converted to jpgs.

If this data has been unzipped to folder `kitti_odom`, a model can be evaluated with:
```shell
python evaluate_pose.py --eval_split odom_9 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
python evaluate_pose.py --eval_split odom_10 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
```


## 📦 Precomputed results

You can download our precomputed disparity predictions from the following links:


| Training modality | Input size  | `.npy` filesize | Eigen disparities                                                                             |
|-------------------|-------------|-----------------|-----------------------------------------------------------------------------------------------|
| Mono              | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192_eigen.npy)           |
| Stereo            | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192_eigen.npy)         |
| Mono + Stereo     | 640 x 192   | 343 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192_eigen.npy)  |
| Mono              | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320_eigen.npy)          |
| Stereo            | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320_eigen.npy)        |
| Mono + Stereo     | 1024 x 320  | 914 MB          | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320_eigen.npy) |



## 👩‍⚖️ License
Copyright © Niantic, Inc. 2019. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
