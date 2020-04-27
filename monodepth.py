# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from cv2 import imshow, VideoCapture, resize, waitKey, destroyAllWindows, putText, FONT_HERSHEY_SIMPLEX, CAP_PROP_FPS

import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from .networks import *

"""
--model_name
"mono_640x192",
"stereo_640x192",
"mono+stereo_640x192",
"mono_no_pt_640x192",
"stereo_no_pt_640x192",
"mono+stereo_no_pt_640x192",
"mono_1024x320",
"stereo_1024x320",
"mono+stereo_1024x320"
"""

model_name = "mono+stereo_640x192"
isinit = False
feed_height = 0
feed_width = 0
device = None
encoder = None
depth_decoder = None


def monodepth_init():
	global isinit
	global feed_width
	global feed_height
	global device
	global encoder
	global depth_decoder
	global gray_detector

	if torch.cuda.is_available():
		device = torch.device("cuda")

	model_path = os.path.join("models", model_name)
	print("[DepthAI_2] Loading model from ", model_path)
	encoder_path = os.path.join(model_path, "encoder.pth")
	depth_decoder_path = os.path.join(model_path, "depth.pth")

	# LOADING PRETRAINED MODEL
	print("[DepthAI_2] Loading pretrained encoder")
	encoder = ResnetEncoder(18, False)
	loaded_dict_enc = torch.load(encoder_path, map_location=device)

	feed_height = loaded_dict_enc['height']
	feed_width = loaded_dict_enc['width']
	filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
	encoder.load_state_dict(filtered_dict_enc)
	encoder.to(device)
	encoder.eval()

	print("[DepthAI_2] Loading pretrained decoder")
	depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

	loaded_dict = torch.load(depth_decoder_path, map_location=device)
	depth_decoder.load_state_dict(loaded_dict)

	depth_decoder.to(device)
	depth_decoder.eval()
	isinit = True

def start_monodepth(frame, img_type="gist_gray"):
	if not isinit:
		raise Exception("[DepthAI_2] monodepth init calistirilmadi.")

	input_image = frame

	original_height, original_width, original_channel = input_image.shape
	input_image = pil.fromarray(resize(input_image, (feed_width, feed_height)))
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)

	# PREDICTION
	input_image = input_image.to(device)
	features = encoder(input_image)
	outputs = depth_decoder(features)

	disp = outputs[("disp", 0)]
	disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

	# showing colormapped depth image
	disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
	vmax = np.percentile(disp_resized_np, 95)
	normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
	mapper = cm.ScalarMappable(norm=normalizer, cmap=img_type)
	return (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

if __name__ == '__main__':
    test_simple()
