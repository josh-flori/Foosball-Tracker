# /users/josh.flori/desktop/test/bin/python3 /users/josh.flori/documents/josh-flori/foosball-tracker/m__process_score_images.py  -v='/users/josh.flori/desktop/foos_gods.h264' -o='/users/josh.flori/desktop/' 




# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_file", required=True,type=str,
	help="path to input video file")    
ap.add_argument("-o", "--output_dir", required=True,type=str,
	help="output path for faces")

    
args = vars(ap.parse_args())



def output_cropped_frames(frame,frame_num):
	cropped = frame[200:600,100:1100]
	cv2.imwrite(args["output_dir"]+"frame_"+str(frame_num)+".jpg", cropped)
 



vidcap = cv2.VideoCapture(args["video_file"])
frame_num = 0
success = True
while success:
        success,color_img = vidcap.read()
        output_cropped_frames(color_img,frame_num)
        frame_num += 1   