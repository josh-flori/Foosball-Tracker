# /users/josh.flori/desktop/test/bin/python3 /users/josh.flori/documents/josh-flori/foosball-tracker/m__find_faces.py -p='/users/josh.flori/desktop/opencv_face_detector.pbtxt.txt' -m='/users/josh.flori/desktop/res10_300x300_ssd_iter_140000.caffemodel' -o='/users/josh.flori/desktop/' -v='/users/josh.flori/a.mp4' -n=3






# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output_dir", required=True,type=str,
	help="output path for faces")
ap.add_argument("-v", "--video_file", required=True,type=str,
	help="path to input video file")    
ap.add_argument("-n", "--num_faces", required=True,type=int,
	help="number of people in the shot")
    
args = vars(ap.parse_args())

# load our serialized model from disk
# print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


def output_faces(frame,frame_num):
    # construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    	(300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    #print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    
    # count the number of faces.
    confidences = [detections[0, 0, i, 2] for i in range(0, detections.shape[2]) if detections[0, 0, i, 2] > args["confidence"]]
    
    # there are many artifact faces detected. sometimes there will be more or less than there should be. in that case, we would be unable to assume proper Y labeling in train_network.py. restricting to the proper number of faces keeps us in line with auto-Y labeling.
    if len(confidences)==args["num_faces"]:
        
        # loop over the detections
        for face in range(0, detections.shape[2]):
        	# extract the confidence (i.e., probability) associated with the
        	# prediction
        	confidence = detections[0, 0, face, 2]
 
        	# filter out weak detections by ensuring the `confidence` is
        	# greater than the minimum confidence
        	if confidence > args["confidence"]:
        		# compute the (x, y)-coordinates of the bounding box for the
        		# object
        		box = detections[0, 0, face, 3:7] * np.array([w, h, w, h])
        		
        		(startX, startY, endX, endY) = box.astype("int")
 
        		# draw the bounding box of the face along with the associated
        		# probability
        		text = "{:.2f}%".format(confidence * 100)
        		y = startY - 10 if startY - 10 > 10 else startY + 10
        		cv2.rectangle(frame, (startX, startY), (endX, endY),
        			(0, 0, 255), 2)
        		cv2.putText(frame, text, (startX, y),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        		cropped = frame[startY:endY, startX:endX]
        		cv2.imwrite(args["output_dir"]+"frame_"+str(frame_num)+"_person_"+str(face)+".jpg", cropped)
 



vidcap = cv2.VideoCapture(args["video_file"])
frame_num = 0
success = True
while success:
        success,color_img = vidcap.read()
        #blkwht_img = np.asarray(Image.fromarray(color_img).convert('L'))
        output_faces(color_img,frame_num)
        frame_num += 1        