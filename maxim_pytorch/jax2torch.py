# -*- coding: utf-8 -*-

#convert pretrained Jax params of MAXIM to Pytorch
import argparse
import collections
import io

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from maxim_torch import MAXIM_dns_3s


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.
    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.
    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.
    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree

def get_params(ckpt_path):
    """Get params checkpoint."""
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params["opt"]["target"]
    return params

def modify_jax_params(flat_jax_dict):
    modified_dict = {}
    for key, value in flat_jax_dict.items():
        key_split = key.split("/")
        modified_value = torch.tensor(value, dtype=torch.float)
        

        #modify values
        num_dim = len(modified_value.shape)
        if num_dim == 1:
            modified_value = modified_value.squeeze()
        elif num_dim == 2 and key_split[-1] == 'kernel':
            # for normal weight, transpose it
            modified_value = modified_value.T
        elif num_dim == 4 and key_split[-1] == 'kernel':
            modified_value = modified_value.permute(3, 2, 0, 1)
            if num_dim ==4 and key_split[-2] == 'ConvTranspose_0' and key_split[-1] == 'kernel':
                modified_value = modified_value.permute(1, 0, 2, 3)


        #modify keys
        modified_key = (".".join(key_split[:]))
        if "kernel" in modified_key:
            modified_key = modified_key.replace("kernel", "weight")
        if "LayerNorm" in modified_key:
            modified_key = modified_key.replace("scale", "gamma")
            modified_key = modified_key.replace("bias", "beta")
        if "layernorm" in modified_key:
            modified_key = modified_key.replace("scale", "gamma")
            modified_key = modified_key.replace("bias", "beta")

        modified_dict[modified_key] = modified_value

    return modified_dict


def main(args):
    jax_params = get_params(args.ckpt_path)
    [flat_jax_dict] = pd.json_normalize(jax_params, sep="/").to_dict(orient="records")  #set separation sign

    # Amend the JAX variables to match the names of the torch variables.
    modified_jax_params = modify_jax_params(flat_jax_dict)

    # update and save
    model = MAXIM_dns_3s()
    maxim_dict = model.state_dict()
    maxim_dict.update(modified_jax_params)
    torch.save(maxim_dict, args.output_file)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the JAX pre-trained MAXIM weights to Pytorch."
    )
    parser.add_argument(
        "-c",
        "--ckpt_path",
        default="maxim_ckpt_Denoising_SIDD_checkpoint.npz",
        type=str,
        help="Checkpoint to port.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="torch_weight.pth",
        type=str,
        help="Output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
