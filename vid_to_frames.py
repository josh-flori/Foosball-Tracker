# The purpoes of this script is to convert a video file from the pi into individual frames.
# These frames will be our labeled training data that our image model learns on.
# Later, we will put that image model onto the pi and use it to perform inference (image recognition) on a video stream in real time.



# EXAMPLE USAGE:
# /users/..../python3 /users/.../vid_to_frames.py  -v='/users/josh.flori/desktop/test.h264' -o='/users/josh.flori/desktop/training_data/' 


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



def output_frames(frame,frame_num):
	# If you want to crop down to a specific part of the video file, you can do that like so:
    # cv2.imwrite(args["output_dir"]+"frame_"+str(frame_num)+".jpg", frame[200:600,100:1100])
	cv2.imwrite(args["output_dir"]+"frame_"+str(frame_num)+".jpg", frame)
 



vidcap = cv2.VideoCapture(args["video_file"])
frame_num = 0
success = True
# This will throw an error when it reaches the last frame and tries to read the next frame since there is no next frame.
while success:
        success,color_img = vidcap.read()
        color_img=normalize(color_img)
        output_frames(color_img,frame_num)
        frame_num += 1   