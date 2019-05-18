# Foosball-Tracker

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
