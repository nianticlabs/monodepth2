## PyTorch re-implementation of MAXIM.
`maxim_torch.py` is the PyTorch re-implementation of 3-stage MAXIM architecture for image denoising. 


`jax2torch.py` is leveraged to convert JAX weights (from a pretrained checkpoint) to PyTorch, and then save it as a dictionary which can be loaded directly to a PyTorch-implemented MAXIM model. To use this script, you should first download the pretrained JAX model from the official directory.

It should be noted that due to the incompatibility between `flax.linen.ConvTranspose` and `torch.nn.ConvTranspose2d`, even if you load exactly the same pretrained parameters, the outputs of JAX model and PyTorch model are not exactly the same, though the difference is small.