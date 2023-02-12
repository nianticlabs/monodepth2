import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--npy_pth', type=str,
                        help='path to npy')
    parser.add_argument('--output_pth',type=str,
                        help='where output is stored',default='./Test_results')

    parser.add_argument('--model_type', type=str,
                        help='model_type', default='original')

    return parser.parse_args()


def npy_to_jpg(args):
    if not os.path.exists(os.path.join(args.output_pth, args.model_type)):
        print('Since output path does not exist, make output folder')
        os.makedirs(os.path.join(args.output_pth, args.model_type))

    npy_list = np.load(args.npy_pth)
    for i in range(len(npy_list)):
        depth_color=_DEPTH_COLORMAP(npy_list[i])[..., :3]
        img=pil.fromarray((depth_color*255).astype(np.uint8))
        img.save(os.path.join(args.output_pth, args.model_type, f'disp_{i}.jpg'))


if __name__ == '__main__':
    args=parse_args()
    npy_to_jpg(args)

