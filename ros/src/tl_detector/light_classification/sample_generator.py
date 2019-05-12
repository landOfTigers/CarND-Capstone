#!/usr/bin/env python

import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


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

    y_one_hot = LabelBinarizer().fit_transform(np.array(tl_states))
    return np.array(images), y_one_hot


def create_train_validation_split():
    samples_list = read_from_log_file('labeled_data.csv')
    train_list, validation_list = train_test_split(samples_list, test_size=0.2, shuffle=True)

    train_samples = create_samples_from_list(train_list)
    validation_samples = create_samples_from_list(validation_list)

    return train_samples, validation_samples


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

    train, validation = create_train_validation_split()
    train_images = np.array(train[0])
    train_labels = np.array(train[1])
    validation_images = np.array(validation[0])
    validation_labels = np.array((validation[1]))
    print(train_images.shape)
    print(train_labels.shape)
    print(validation_images.shape)
    print(validation_labels.shape)

    # display one image for demo purposes
    plt.imshow(cv2.cvtColor(train_images[3], cv2.COLOR_BGR2RGB))
    plt.show()
