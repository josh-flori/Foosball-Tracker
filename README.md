# Foosball-Tracker, Overview of tasks (in progress)

#### 1) SCORE TRACKER
Have two cameras pointing at the goals, keeping track of the score. Train network to recognize the points, taking a picture every second in real play. Write a script to keep track of what the current score is. When game over, send signal to voice unit to give some message. Custom messages like, if it was really long, or even saying congrats to the winners by name.. that would be cool.
#### 2) SAVE DATA TO TABLE
Get working with a database, saving a log of the information, players, score, game duration in minutes, length until first goal, longest time between goals, date.
#### 3) RECOGNIZE FACES 
Get facial recognition working in order to log the players. 
#### 4) TRIGGER RECORDING, LAUNCH ALL PROCESSES
Either have trigger word to start recording, or have them push a button... in either case have the voice unit announce that it is recording, probably have it announce when it is done as well.
#### 5) RECORD TABLE PLAY, SAVE KEY EVENTS
Have camera pointing at table recording gameplay. Have voice unit record audio. When game is completed, look for key moments in audio, save those video clips as notable events, put videos online somehow.


GET ANOTHER VISION KIT
FIGURE OUT HOW TO CONTROL THEM WITH A SINGLE UNIT

1)
* Take a bunch of video of the scores incrimenting, including hand frames
* Annotate data with score position
* Train a model to correctly classify that information with perfect accuracy. If not perfect, figure out code way to allow for it.
* Convert model to tflite format, put on pi.
* Write a script for the pi that uses tflite network and takes a photo every second or something, then sends that information back to a central script for processing.
* Write a centralized script that takes as input those signals and does xyz with that information

2) 
* Figure out how to get a database set up
* Figure out how to write data from python to the table
* Write a script whose only purpose is to write data
* Write a script whose purpose is to take the data collected from ALL the other scripts and dump it to table
* This process should be done last.

3) 
* Figure out how to handle the addition of players
* Train network on a sample of players to start
* Try to get enough data to be able to use a hand built network
* Test pretrained model on dummy data just to see how performance compares to what I already have set up,,,, just to establish a baseline
* Use pretrained model if needed
* Write script to record every one second or something, process, then send the information back to a monitoring script on the central unit to write the player data to the table, maybe take it from video feed, that way you could compile player reactions and do cool edits 

4)
* Write script to look for trigger word and print something to the console
* Then when all other scripts are compiled, figure out way to trigger other scripts 

5)
* Write script triggered by trigger word, record table play
* Write script to record audio of gameplay
* Look for peaks in audio waveforms and manually check if that is good enough to find game highlights
* Split audio into speakers, then look for keywords
* From keyword timestamp, split video, get n seconds beforehand
* Save to table
