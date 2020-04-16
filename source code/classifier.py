from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# functions
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def print_layer_trainable(model):
    for layer in model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def save_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    # plt.show()
    
    # Save figure 
    plt.savefig(save_dir+"result.png")
    
# define parameter
is_use_gpu = 1
gpu_id = '1'
max_gpu_mem = 1

if is_use_gpu:
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
  device = '/device:GPU:0'
else:
  device = "/cpu:0"

batch_size = 32
save_to_dir = './cropped_resized_trans_queries/'
train_dir = './cropped_resized_queries/'
test_dir = './cropped_resized_queries/'
save_dir = './save/'

epoch = 40
step_per_epoch = 10

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = max_gpu_mem
config.gpu_options.allow_growth=True
config.allow_soft_placement = True
config.log_device_placement=True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# load the model
model = VGG16(include_top=True, weights='imagenet')

# get model input shape
input_shape = model.layers[0].output_shape[0][1:3]

# prepare data
train_data_generator = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')

test_data_generator = ImageDataGenerator(rescale=1./255)

# generate image for training
train_data_loader = train_data_generator.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)
# generate image for testing
test_data_loader = test_data_generator.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

# get data information
test_step = test_data_loader.n / batch_size

train_img_path = path_join(train_dir, train_data_loader.filenames)
test_img_path = path_join(test_dir, test_data_loader.filenames)

train_cls = train_data_loader.classes
test_cls = test_data_loader.classes

cls_name = list(train_data_loader.class_indices.keys())
n_cls = train_data_loader.num_classes

cls_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(train_cls),
                                    y=train_cls)

# Transfer learning
transfer_layer = model.get_layer('block5_pool')
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

# Make pre-trained kernel trainable
conv_model.trainable = True

# Init model
new_model = Sequential()

# Add the convolutional part of the VGG16 model
new_model.add(conv_model)

# Flatten the feature map
new_model.add(Flatten())

# Dense layer.
new_model.add(Dense(1024, activation='relu'))

# Dropout-layer to prevent overfitting
new_model.add(Dropout(0.5))

# Add softmax layer for classification
new_model.add(Dense(n_cls, activation='softmax'))

optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

# compile to apply the setting
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

checkpoint = ModelCheckpoint(save_dir+"model.h5", monitor='categorical_accuracy', verbose=1,
    save_best_only=True, mode='auto', save_weights_only=False, period=1)

history = new_model.fit_generator(generator=train_data_loader,
                                  epochs=epoch,
                                  steps_per_epoch=step_per_epoch,
                                  class_weight=cls_weight,
                                  validation_data=test_data_loader,
                                  validation_steps=test_step,
                                  callbacks=[checkpoint]
                                  )
save_training_history(history)

# Save the model
new_model.save(save_dir+'model.h5')


