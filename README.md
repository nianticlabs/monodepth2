# G2S

This is a reference implementation for using G2S loss described in the [ICRA 2021 paper](https://ieeexplore.ieee.org/document/9561441)

> **Multimodal Scale Consistency and Awareness for Monocular Self-Supervised Depth Estimation**
>
>  by [Hemang Chawla](https://scholar.google.com/citations?user=_58RpMgAAAAJ&hl=en&oi=ao), [Arnav Varma](https://scholar.google.com/citations?user=3QSih2AAAAAJ&hl=en&oi=ao), [Elahe Arani](https://www.linkedin.com/in/elahe-arani-630870b2/) and [Bahram Zonooz](https://scholar.google.com/citations?hl=en&user=FZmIlY8AAAAJ).

in the Monodepth2 repository for KITTI Eigen Zhou split.

The official code is available [here](https://github.com/NeurAI-Lab/G2S).

This code is for non-commercial use following the original license from Monodepth2; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{chawlavarma2021multimodal,
	author={H. {Chawla} and A. {Varma} and E. {Arani} and B. {Zonooz}},
	booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
	title={Multimodal Scale Consistency and Awareness for Monocular Self-Supervised
	Depth Estimation},
	location={Xi’an, China},
	publisher={IEEE (in press)},
	year={2021}
}
```


## ⏳ Training


**Monocular training:**
```shell
python train.py --model_name g2s --data_path /path/to/KITTI/raw_data/sync --log_dir /path/to/log/dir/ --g2s --png (if images are in png)
```

## 👩‍⚖️ License
Please see the [license file](LICENSE) for terms.
