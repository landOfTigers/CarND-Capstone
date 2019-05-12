#!/usr/bin/env python

import datetime
import json

from keras import backend as K
from keras.layers import Lambda, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

from sample_generator import create_augmented_training_set

# 1: define model architecture
model = Sequential()
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (150, 200)), input_shape=(600, 800, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='mse', optimizer='adam')
model.summary()

# 2: compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
x_train, y_train = create_augmented_training_set()
start_time = datetime.datetime.now()
history = model.fit(x_train, y_train, epochs=4, batch_size=16, shuffle=True, validation_split=0.2)
training_time = str((datetime.datetime.now() - start_time).seconds)

# 3: save history to file
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H:%M')
with open('model_history/%s.json' % timestamp, 'w') as f:
    statistics = {'training_time_seconds': training_time}
    statistics.update(history.history)
    json.dump(statistics, f)
    model.save('model.h5')
