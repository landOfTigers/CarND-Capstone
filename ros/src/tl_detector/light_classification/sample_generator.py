#!/usr/bin/env python

import csv

import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def read_from_log_file(file_name):
    samples = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
        samples = samples[1:]  # throw away header line
    return samples


def flip_and_stack_training_data(xx, yy):
    x_flipped = []
    y_flipped = []
    for x, y in zip(xx, yy):
        x_flipped.append(np.fliplr(x))
        y_flipped.append(y)

    x_result = np.vstack((np.array(xx), np.array(x_flipped)))
    y_result = np.append(np.array(yy), np.array(y_flipped))

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

    x_train, y_train = flip_and_stack_training_data(images, tl_states)
    y_one_hot = LabelBinarizer().fit_transform(y_train)

    return x_train, y_one_hot


if __name__ == '__main__':
    x_samples, y_samples = create_samples_from_log()
    print(x_samples.shape)
    print(y_samples.shape)
