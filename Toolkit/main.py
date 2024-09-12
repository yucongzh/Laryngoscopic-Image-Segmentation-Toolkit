from Utils import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import sys
import os
from skimage.io import imread
from segment_anything import sam_model_registry

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--vpath',       default='./input/00001_rgb.png',    type=str)
parser.add_argument('--rootdir',     default="./",                       type=str)
parser.add_argument('--gpu',         default="0",                        type=str)
args = parser.parse_args()

#%% Basic settings
vid = os.path.splitext(os.path.basename(args.vpath))[0]
vid2fpath = {vid:args.vpath}
vids = list(vid2fpath.keys())
os.system("export VERBOSE=False")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') # or cpu
width=256; height=256 # image size

#%% Load YOLO-v5 for larynx detection
chkpt_path = f"{args.rootdir}/checkpoints/yolov5_model.pt"
yolo_model = torch.hub.load(f"{args.rootdir}/models/yolov5", 'custom', path=chkpt_path, source='local', _verbose=False)
yolo_model = yolo_model.to(device)
yolo_model.eval()

#%% Load segmentation models
# load U-Net for glottis segmentation
from Models.unet_model.unet import UNet
chkpt_path = f"{args.rootdir}/checkpoints/best_model.dict"
net = UNet(n_channels=1, n_classes=1).to(device)
net.load_state_dict(torch.load(chkpt_path, map_location=device)['model_state_dict'])
net.eval()

# load Segment Anything Model for vocal fold segmentation
sam_checkpoint = f"{args.rootdir}/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


#%%
if __name__ == "__main__":
    for vid in vids:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        img = imread(args.vpath)
        gmask,_,bbox,mid,angle = peek_v2(img, net, yolo_model, device)
        vf_mask = get_sam_masks_unet_with_yolo_lr_v7(img,[],sam,gmask,bbox,mid,angle,gmask,yolo_model)
        vf_mask = np.uint8(vf_mask)
        plt.imsave(f"{args.rootdir}/output/{vid}_mask.png", vf_mask)
        print('Message-Segmentation Done.')
