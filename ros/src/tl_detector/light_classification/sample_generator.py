#!/usr/bin/env python

import csv

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
IMAGE_CHANNEL = 3
NUM_CLASSES = 4


def read_from_log_file(file_name):
    samples = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
        samples = samples[1:]  # throw away header line
    return samples


def create_samples_from_list(samples_list):
    images = []
    tl_states = []
    for line in samples_list:
        # add images
        image = cv2.imread(line[0])
        images.append(image)

        # add traffic light states
        tl_state = int(line[1])
        tl_states.append(tl_state)

    y_one_hot = to_categorical(np.array(tl_states), num_classes=NUM_CLASSES)
    return np.array(images), y_one_hot


def create_train_validation_samples_lists():
    samples_list = read_from_log_file('labeled_data.csv')
    train_list, validation_list = train_test_split(samples_list, test_size=0.2, shuffle=True)
    return train_list, validation_list


def raw_training_data_generator(samples_list, batch_size):
    num_samples = len(samples_list)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples_list = samples_list[offset:offset + batch_size]
            x_train, y_train = create_samples_from_list(batch_samples_list)
            yield x_train, y_train


def create_augmented_data_generator(samples_list, batch_size):
    raw_gen = raw_training_data_generator(samples_list, batch_size)
    augmented_image_generator = ImageDataGenerator(horizontal_flip=True, rotation_range=5, width_shift_range=0.05,
                                                   height_shift_range=0.05, zoom_range=[0.95, 1.05],
                                                   fill_mode='reflect', data_format='channels_last')
    for x_samp, y_samp in raw_gen:
        aug_gen = augmented_image_generator.flow(x_samp, y_samp, batch_size=x_samp.shape[0])
        x_samp, y_samp = next(aug_gen)
        yield x_samp, y_samp


def print_samples_stats():
    print("Statistic of original (non-augmented) data set:")
    samples_list = read_from_log_file('labeled_data.csv')

    tl_states = []
    for line in samples_list:
        tl_state = int(line[1])
        tl_states.append(tl_state)

    total = len(tl_states)
    red = tl_states.count(0)
    yellow = tl_states.count(1)
    green = tl_states.count(2)
    unknown = tl_states.count(3)

    output = '''
        Total number of traffic lights: {}
        Number of red lights: {}
        Number of yellow lights: {}
        Number of green lights: {}
        Number no/unknown lights: {}
        '''.format(total, red, yellow, green, unknown)

    print (output)


if __name__ == '__main__':
    print_samples_stats()

    samp_list = read_from_log_file('labeled_data.csv')[:16]
    x, y = create_samples_from_list(samp_list)
    print(x.shape)
    print(y.shape)
