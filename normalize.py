# EXAMPLE USAGE:
# /users/..../python3 /users/.../vid_to_frames.py  -v='/users/josh.flori/desktop/test.h264' -o='/users/josh.flori/desktop/training_data/' 


import numpy as np
import argparse
import cv2
import os


# Normalize does the following for each channel: image = (image - mean) / std
# https://discuss.pytorch.org/t/understanding-transform-normalize/21730
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,type=str,
	help="output path for faces")

    
args = vars(ap.parse_args())



for folder in os.listdir(args["input_dir"]):
    if '.DS_Store' not in folder:
        for image in os.listdir(args["input_dir"]+folder):
            if '.DS_Store' not in image:
                normalized=cv2.imread(args["input_dir"]+folder+"/"+image).astype(np.float32)
                # normalized = cv2.normalize(normalized, None,norm_type=cv2.NORM_MINMAX)
                normalized = (normalized - 128) / 128
                cv2.imwrite(args["input_dir"]+folder+"/"+image,normalized)