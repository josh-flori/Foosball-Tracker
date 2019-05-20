# scp pi@192.168.11.2:~/AIY-projects-python/src/examples/vision/faces.jpg /users/josh.flori/desktop
# scp pi@192.168.11.2:~/foos_gods.h264 /users/josh.flori/desktop


import picamera
import datetime
import threading
import subprocess
camera = picamera.PiCamera()
camera.framerate = 60  ## 120 is the highest possible framerate and will not work beyond a certain resolution
#camera.iso = 1600
camera.resolution = (1280, 720)  ## please note 900x200 refers to the number of columns (width) and then the height (rows) in pixels. all other references to matrices and arrays in python and excel will be the exact opposite which means that as an array, a 900x200 image is actually 200x900 (rows by columns) as opposed to columns by rows. 
camera.start_preview()
camera.start_recording('foos_gods.h264')
camera.stop_recording()


conv = subprocess.Popen(convCommand, shell = True)
convCommand = ['MP4Box', '-add', 'foos_gods.h264', '-o', FILEOUT + 'foos_gods.mp4']


//---------------------------------------------------
//-----------  FACE_CAMERA_TRIGGER .PY  ------------
//---------------------------------------------------

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection

from picamera import PiCamera


def main():
    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (1640, 922)  # Full Frame, 16:9 (Camera v2)
        camera.start_preview()

        # Do inference on VisionBonnet
        i=1
        with CameraInference(face_detection.model()) as inference:
            for result in inference.run():
                if len(face_detection.get_faces(result)) >= 1:
                    camera.capture('faces'+str(i)+'.jpg')
                    i+=1

        # Stop preview
        camera.stop_preview()



//---------------------------------------------------
//-------------   FACE_DETECTION .py   --------------
//---------------------------------------------------


import argparse

from PIL import Image, ImageDraw

from aiy.vision.inference import ImageInference
from aiy.vision.models import face_detection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input', required=True)
    parser.add_argument('--output', '-o', dest='output')
    args = parser.parse_args()

    with ImageInference(face_detection.model()) as inference:
        image = Image.open(args.input)
        draw = ImageDraw.Draw(image)
        faces = face_detection.get_faces(inference.run(image))
        for i, face in enumerate(faces):
            print('Face #%d: %s' % (i, face))
            x, y, width, height = face.bounding_box
            draw.rectangle((x, y, x + width, y + height), outline='red')
        if args.output:
            image.save('~/i.jpg')
