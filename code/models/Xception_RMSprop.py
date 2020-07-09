#TRASNFER LEARNING WITH A PRETRAINED CONVNET

#Inports general
import os
import numpy as np
import matplotlib.pyplot as plt

#tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

## Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
NODES_HIDDEN_0 = 512
BASE_TRAINABLE = False
PATIENCE = 3
VALIDATION_STEPS = 20

#DATA PREPROCESSING

#Data download

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str


for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
#Format the data

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

#Apply this to each item on the dataset using map method
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#Shuffle and batch the data
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#Inspect a batch of data
for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape


early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',    # Stop training when `val_loss` is no longer improving
        min_delta=1e-3,               # "no longer improving" being defined as "no better than 0 less"
        patience=PATIENCE,         # "no longer improving" being further defined as "for at least 3 epochs"
        verbose=0,                 # Quantity of printed output
        mode='max',                # In 'max' mode, training will stop when the quantity monitored has stopped increasing;
        #baseline=None,
        #restore_best_weights=False
        )



#CREATE THE BASE MODEL OF THE PRETRAINED CONVNET
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Join list of required callbacks
callbacks = [early_stopping_callback]


# Create the base model from the pre-trained model Xception
base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features.
feature_batch = base_model(image_batch)
#print(feature_batch.shape)

#FEATURE EXTRACTION

#Freeze the convolutional base
base_model.trainable = BASE_TRAINABLE
# Let's take a look at the base model architecture
#base_model.summary()

#ADD A CLASSIFICATION HEAD
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
#print(feature_batch_average.shape)

#Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image.
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#Now stack the feature extractor, and these two layers using a tf.keras.Sequential model:
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

#COMPILE THE MODEL
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#TRAIN THE MODEL


loss0,accuracy0 = model.evaluate(validation_batches, steps = VALIDATION_STEPS)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches, callbacks=callbacks,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

#LEARNING CURVES
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()














