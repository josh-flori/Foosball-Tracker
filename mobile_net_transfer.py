# https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
# https://www.tensorflow.org/alpha/guide/saved_model

from tensorflow.keras import backend as K
from tensorflow.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam



base_model=tf.keras.applications.MobileNet(weights='imagenet',include_top=False) 
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    
    
model=Model(inputs=base_model.input,outputs=preds)
for i,layer in enumerate(model.layers):
  print(i,layer.name)
  
for layer in model.layers[:88]:
    layer.trainable=False
for layer in model.layers[88:]:
    layer.trainable=True  
    
    
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('C:/Users/Ferhat/Python Code/Workshop/Tensoorflow transfer learning/downloads',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)    
    
    
    
    
    
    
    
    
    
    
    
    
    
preprocessed_image = prepare_image('/users/josh.flori/desktop/Screen Shot 2019-05-20 at 10.22.52 AM.png')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results
    
Image(filename='/users/josh.flori/desktop/frame_264.jpg')
