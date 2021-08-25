import os
import numpy as np
import time
import PIL.Image as pil
import torch
from torchvision import transforms, datasets
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist, monodepth2_models_path

MODEL_NAMES = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320"
]

class monodepth2:

    def __init__(self, model_name=MODEL_NAMES[2], no_cuda=False, pred_metric_depth=False) -> None:
        assert model_name in MODEL_NAMES, "Invalid Model Name"
        
        if torch.cuda.is_available() and not no_cuda:
            self.device = torch.device("cuda")
            print("GPU Visible")
        else:
            self.device = torch.device("cpu")
            print("GPU not visible; CPU mode")
        
        if pred_metric_depth and "stereo" not in model_name:
            print("Warning: The pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")
        
        download_model_if_doesnt_exist(model_name=model_name)
        model_path = os.path.join(monodepth2_models_path, model_name)

        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        self.depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        #torch.no_grad()
        pass

    def eval(self, input_image):
        
        with torch.no_grad():
            # Load image and preprocess
            
            input_image = pil.fromarray(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)


            #disp_resized_np = disp_resized.squeeze().cpu().numpy()
            disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

        return colormapped_im
        

if __name__=="__main__":
    # Webcam depth
    import cv2
    cap = cv2.VideoCapture(0)
    m = monodepth2()

    plt.ion()
    # plt.draw()
    plt.show(block=False)

    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    while(True):
        try:
            # Capture the video frame
            # by frame
            ret, frame = cap.read()

            #frame = cv2.imread('monodepth2/assets/test_image.jpg')
        
            # Display the resulting frame
            depth = m.eval(frame)

            ax1.imshow(frame)
            ax2.imshow(depth)
            
            #cv2.imwrite('tmps/frame.png', frame)
            #cv2.imwrite('tmps/depth.png', depth)
            #depth.save('depth.jpeg')
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            #time.sleep(0.01)
            plt.pause(0.001)
        except:
             break
    
    plt.show()
    
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
