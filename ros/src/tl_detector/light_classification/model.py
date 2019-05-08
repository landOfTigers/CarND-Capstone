#!/usr/bin/env python

import json
import datetime
from keras.layers import Lambda, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

from sample_generator import create_samples_from_log

# 1: define model architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(600, 800, 3)))
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
x_train, y_train = create_samples_from_log()
history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# 3: save history to file
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H:%M')
with open('model_history/%s.json' % timestamp, 'w') as f:
    json.dump(history.history, f)

model.save('model.h5')
