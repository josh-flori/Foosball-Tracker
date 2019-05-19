# THE PURPOSE OF THIS SCRIPT IS TO TRAIN A NETWORK TO RECOGNIZE OUR FACES
# /users/josh.flori/desktop/test/bin/python3 /users/josh.flori/documents/josh-flori/foosball-tracker/m__train_network.py -i='/users/josh.flori/desktop/' -f 0 1

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import argparse
import matplotlib.pyplot as plt

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--path_to_images", required=True,type=str,
	help="path to input images")
ap.add_argument('-f', '--faces_to_text', nargs='*', default=[],help="a text list of the names of the faces corresponding to the 0th:nth faces represented in the images to be loaded") # # find_faces.py outputs naming structure like: "frame_0_person_0", so if person 0 was "bob" then your list would be ["bob"] and so on for all faces)
ap.add_argument('-w', '--path_to_model',type=str, default='',help='if first time training model, this should be the path to where you want to save you model, else, path to where you already saved your model.')
ap.add_argument('-f', '--first_time_training_model',type=str, default='',help="should be either (y,n), determines whether we load existing model weights or not.")

    

args = vars(ap.parse_args())


def getfiles(dirpath):
    images = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s)) and "jpg" in s]
    images.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return images




#####################
#    LOAD IAMGES    #
#####################
images=getfiles(args["path_to_images"])

    
X = np.array([cv2.resize(cv2.imread(args["path_to_images"]+image),(50,50)) for image in images])
Y = np.array([args["faces_to_text"][i] for i in range(len(args["faces_to_text"]))]*int((len(images)/len(args["faces_to_text"]))))

print(X.shape)
print(Y.shape)
print(len(images))

##########################
#    STANDARDIZE DATA    #
##########################
X = X / 255

##########################
#   ASSERTION CHECKING   #
##########################
assert (X.shape[0] == len(images))
assert (Y.shape[0] == len(images))
print(X.shape)
print(Y.shape)

##########################
#     VISUALIZE DATA     #
##########################
# for i in range(10):
#     print(Y[i])
#     plt.figure()
#     plt.imshow(X[i])
#     plt.show()

#####################
#     RANDOMIZE     #
#####################
np.random.seed(0)
np.random.shuffle(X)
np.random.seed(0)
np.random.shuffle(Y)

#######################
#      TRAIN TEST     #
#######################
train_X, dev_X, test_X = np.split(X, [int(.8 * X.shape[0]), int(.9 * X.shape[0])])
train_Y, dev_Y, test_Y = np.split(Y, [int(.8 * X.shape[0]), int(.9 * X.shape[0])])

########################
#     DEFINE MODEL     #
########################

if args["first_time_training_model"].lower()=="y":
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X[0].shape[0], X[0].shape[1], 3)),
        keras.layers.Dense(50, activation=tf.nn.sigmoid),
        keras.layers.Dense(50, activation=tf.nn.sigmoid),
        keras.layers.Dense(50, activation=tf.nn.sigmoid),
        keras.layers.Dense(len(args['faces_to_text']), activation=tf.nn.softmax)
    ])

    ########################
    #     COMPILE MODEL    #
    ########################
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
else:
    model = tf.keras.models.load_model(args["path_to_model"])
    


#####################
#      FIT MODEL    #
#####################
history = model.fit(train_X, train_Y, epochs=500, verbose=2, validation_data=(dev_X, dev_Y))
history_dict = history.history
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']


def plot_diagnostics(train_acc,val_acc,train_loss,val_loss,which="accuracy"):
    epochs = range(1, len(train_acc) + 1)
    if which=="loss":
        plt.plot(epochs, train_loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
    else:
        plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.ylabel('Accuracy')
        
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()



plot_diagnostics(train_acc,val_acc,train_loss,val_loss,"loss")
plot_diagnostics(train_acc,val_acc,train_loss,val_loss,"accuracy")



################################
#     EVALUATE ON TEST DATA    #
################################
model.evaluate(test_X, test_Y)

#########################################
#   PRODUCE THE PREDICTIONS, IF NEEDED  #
#########################################
predictions = model.predict(test_X)
predictions.shape
np.argmax(predictions[0])


save_model=input("would you like to save this model (weights, state) and overwrite any previously saved weights? (y/n)")
if save_model.lower()=="y":
    model.save(args["path_to_model"])




