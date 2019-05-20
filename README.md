# Foosball-Tracker

## Things to do....
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




We have a Foosball table at work. Often, there are extraordinary plays that we would like to record and play back later. And we would also like to track stats like: 

"was this the highest scoring game?" and 

"was this the longest game?" and 

"was this the longest we have ever gone before scoring a point?" and 

"who has the most shots on goal? etc…

Using the google vision kit and google voice kit, we can create a smart unit to track all of those stats as well as use a speaker to interact with and announce important events to the players in real time.

To connect to a pi from terminal, I didn't follow the instructions from google. The IP didn't work. I had to load the boot drive, create an empty file called "ssh" then plug in the pi, connect to computer and in terminal do:

    ssh pi@192.168.11.2

with password being 

    raspberry


