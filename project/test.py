import argparse
import torch
from torch.utils.data import DataLoader

from project.ModelingUtils import test, visualize_disparity_maps
from project.train import make_dir_if_not_exists
from dataset_interface import MyDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-save-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)

    parser.add_argument('--visualize-dir', type=str,
                        help="location of directory to save visualized results to")

    return parser.parse_args()


def main():
    args = get_args()

    model = torch.load(f"{args.model_save_dir}/best.pt")
    test_loader = DataLoader(dataset=MyDataset("test"), batch_size=args.batch_size)
    test(test_loader, model)

    if args.visualize_dir is not None:
        make_dir_if_not_exists(args.visualize_dir)

        # TODO: should eventually be test loader, not train loader
        viz_loader = DataLoader(dataset=MyDataset("train"), batch_size=args.batch_size)

        save_path = f"{args.visualize_dir}/viz.png"
        visualize_disparity_maps(viz_loader, model, save_path)


if __name__ == "__main__":
    main()
