# hmmm important ... https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md

# https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
# https://www.tensorflow.org/alpha/guide/saved_model
# https://keras.io/preprocessing/image/

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import os


# This saves weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format
checkpoint_path = "/users/josh.flori/desktop/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=False,
    # Save weights, every 5-epochs.
    save_freq=1)

logdir="/users/josh.flori/desktop/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# model = tf.keras.Sequential([
#   base_model,
#   tf.keras.layers.GlobalAveragePooling2D(),
#   tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.2),activation='relu'),
#   tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.2),activation='relu'),
#   tf.keras.layers.Dense(3, activation='softmax')
# ])


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(35,110, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(11, activation='softmax'))


# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(
#     x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
# x = Dense(1024, activation='relu')(x)  # dense layer 2
# x = Dense(512, activation='relu')(x)  # dense layer 3
# preds = Dense(3, activation='softmax')(x)  # final layer with softmax activation
#
# model = Model(inputs=base_model.input, outputs=preds)
#
# turn_off(87)

# for i,layer in enumerate(model.layers):
#   print(i,layer.trainable,layer.name)


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1. / 255,
                                   validation_split=0.1)  # set validation split

train_generator = train_datagen.flow_from_directory(
    '/users/josh.flori/desktop/training_data/',
    target_size=(35,110),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    )  # set as training data

validation_generator = train_datagen.flow_from_directory(
    '/users/josh.flori/desktop/training_data/',
    target_size=(35,110),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    )  # set as validation data

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# initialize checkpoint directory
model.save_weights(checkpoint_path.format(epoch=0))


history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // train_generator.batch_size,
    epochs=10, callbacks = [cp_callback,tensorboard_callback])

# Get diagnostic data from history    
history_dict = history.history
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']


def plot_diagnostics(train_acc, val_acc, train_loss, val_loss, which="accuracy"):
    epochs = range(1, len(train_acc) + 1)
    if which == "loss":
        plt.plot(epochs, train_loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
    else:
        plt.plot(epochs, train_acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


plot_diagnostics(train_acc, val_acc, train_loss, val_loss, "loss")
plot_diagnostics(train_acc, val_acc, train_loss, val_loss, "accuracy")


  
  
tf.saved_model.save(model, "/users/josh.flori/11")
  










from tensorflow.keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)
from tensorflow.keras.models import load_model
model = load_model('/users/josh.flori/desktop/model.h5')
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]



from tensorflow.keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])



tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)






converter = tf.lite.TFLiteConverter.from_saved_model('/users/josh.flori/1/')

from tensorflow.keras.models import load_model
m = load_model('/users/josh.flori/11/saved_model.pb')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)













