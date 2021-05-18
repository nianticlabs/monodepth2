---
name: Training on custom training data
about: Problems training on data which isn't KITTI
title: ''
labels: custom-dataset
assignees: ''

---

Thanks for trying out monodepth2 on a different dataset!

Before logging your issue, please look through the issues already tagged with [custom dataset](https://github.com/nianticlabs/monodepth2/issues?q=label%3Acustom-dataset+)  , as your problem may well have been answered there.

In particular please note that:

1. You should know the intrinsics for your custom dataset, and you should set them *normalized* in your dataloader. More details are in the [dataloader comments](https://github.com/nianticlabs/monodepth2/blob/master/datasets/kitti_dataset.py#L24).

2. KITTI is captured at 10 frames per second, and this seems to work well for training in our repo. Most videos are higher frame rate than this, so you may want to consider temporally downsampling your dataset (or setting `--frame_ids` appropriately)

3. If you have moving objects in your training you are likely to still see ‘holes’ punched in your predicted depths, even with automasking turned on (https://github.com/nianticlabs/monodepth2/issues/310) 

4. Monodepth2 is unlikely to work on monocular training data from indoor environments, or captured from difficult camera motions. It is best suited to driving scenarios or other simple forward camera motions.

If you still have issues training, we might be able to help. Please report:

1. What you have changed to create your custom dataset
2. Some example training images
3. What do the predicted depths in the tensorboard output look like?
