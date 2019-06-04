#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trigger PiCamera when face is detected."""




# scp pi@192.168.11.2:~/AIY-projects-python/src/examples/vision/faces.jpg /users/josh.flori/desktop

#scp /users/josh.flori/documents/josh-flori/foosball-tracker/test.py  pi@192.168.11.2:~/foos_gods/test.py

#/home/pi/foos_gods/test.py

#//---------------------------------------------------
#//-----------  FACE_CAMERA_TRIGGER .PY  ------------
#//---------------------------------------------------
# change directory so we can import modules normally
import os
os.chdir('/home/pi/AIY-projects-python/src/examples/vision')
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection

from picamera import PiCamera

# images will be output here
if not os.path.exists('/home/pi/facial_recognition/'):
    os.makedirs('/home/pi/facial_recognition/')

def main():
    print("1")
    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (1640, 922)  # Full Frame, 16:9 (Camera v2)
        camera.start_preview()
#ImageInference
        # Do inference on VisionBonnet
        i=1
        with CameraInference(face_detection.model()) as inference:
            for result in inference.run():
                if len(face_detection.get_faces(result)) >= 1:
                    camera.capture('faces'+str(i)+'.jpg')
                    i+=1

        # Stop preview
        camera.stop_preview()

if __name__ == '__main__':
    main()









class Room(object):
    # Note that we're taking an argument besides self, here.
    def __init__(self, name,color):
        self.name = name  # Set the room's name to the name we got.
        self.color=color
    @staticmethod    
    def update_color(color):
        color=color
        print(color)
    
    
class WhiteRoom(Room):  # A white room is a kind of room.
    def __init__(self):
        # All white rooms have names of 'White Room'.
        self.name = 'White Room'
    def interact(self, line):
        global current_room, white_room,red_room
        if 'white' in line:
            # We could create a new WhiteRoom, but then it
            # would lose its data (if it had any) after moving
            # out of it and into it again.
            current_room = white_room    
        if "red" in line.lower():
            print(True)
            current_room=red_room
            
                
    
class RedRoom(Room):  # A red room is also a kind of room.
    def __init__(self):
        self.name = 'Red Room'
    def interact(self, line):
        global current_room, white_room,red_room
        if 'white' in line:
            # We could create a new WhiteRoom, but then it
            # would lose its data (if it had any) after moving
            # out of it and into it again.
            current_room = white_room    
        if "red" in line.lower():
            current_room=red_room
    
    
def play_game():
    global current_room
    while True:
        line = input(current_room.name + '> ')
        current_room.interact(line)
        
        
        
        
        
        
        
def reset_game():
    global current_room, white_room, red_room
    white_room = WhiteRoom()
    red_room = RedRoom()
    current_room = white_room
    
    
iterator=(i for i in range(1,4))    
[[x*y for y in iterator] for x in iterator]    
    
            
    