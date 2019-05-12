#!/usr/bin/env python

import csv
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from sklearn.preprocessing import LabelBinarizer


def read_from_log_file(file_name):
    samples = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
        samples = samples[1:]  # throw away header line
    return samples


def append_flipped_training_data(xx, yy):
    x_result = list(xx)
    y_result = list(yy)
    for x, y in zip(xx, yy):
        x_result.append(np.fliplr(x))
        y_result.append(y)
    return x_result, y_result


def append_randomly_rotated_training_data(xx, yy):
    random_degree = random.uniform(-5, 5)
    x_result = list(xx)
    y_result = list(yy)
    for x, y in zip(xx, yy):
        rot = sk.transform.rotate(x, random_degree, preserve_range=True).astype(np.uint8)
        x_result.append(rot)
        y_result.append(y)
    return x_result, y_result


def create_samples_from_log():
    samples = read_from_log_file('labeled_data.csv')
    images = []
    tl_states = []
    for line in samples:
        # add images
        image = cv2.imread(line[0])
        images.append(image)

        # add traffic light states
        tl_state = int(line[1])
        tl_states.append(tl_state)
    return images, tl_states


def create_augmented_training_set():
    images, tl_states = create_samples_from_log()
    x_train, y_train = append_flipped_training_data(images, tl_states)
    x_train, y_train = append_randomly_rotated_training_data(x_train, y_train)
    y_one_hot = LabelBinarizer().fit_transform(np.array(y_train))
    return np.array(x_train), y_one_hot


def print_samples_stats():
    print("Statistic of original (non-augmented) data set:")
    images, tl_states = create_samples_from_log()
    total = len(tl_states)
    red = tl_states.count(0)
    yellow = tl_states.count(1)
    green = tl_states.count(2)
    unknown = tl_states.count(4)

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
    x_samples, y_samples = create_augmented_training_set()

    # display one image for demo purposes
    plt.imshow(cv2.cvtColor(x_samples[3], cv2.COLOR_BGR2RGB))
    plt.show()
