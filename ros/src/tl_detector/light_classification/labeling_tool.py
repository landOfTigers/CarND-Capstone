#!/usr/bin/env python

import sys
import termios
import tty
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

IMG_DIR = 'tl_images'
LABELED_DATA_FILE = "labeled_data.csv"


def state_mapper(state):
    if state == "r":
        return 0
    if state == "y":
        return 1
    if state == "g":
        return 2
    return 4


def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def user_input():
    while True:
        print("Select traffic light state: (r)ed, (y)ellow, (g)reen, (u)nknown, (e)xit: ")
        state = get_char().lower()
        if state in ("r", "y", "g", "u", "e"):
            return state
        print("Please select a valid label!")


if __name__ == '__main__':
    plt.ion()
    file = open(LABELED_DATA_FILE, "a")
    img_paths = Path(IMG_DIR).glob('**/*.png')
    tl_state = None
    for img_path in img_paths:
        path = str(img_path)

        # check if image has already been labeled
        if path in open(LABELED_DATA_FILE).read():
            continue

        image = cv2.imread(path)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(0.05)

        tl_state = user_input()
        if tl_state == "e":
            break
        file.write("%s, %s\n" % (path, (state_mapper(tl_state))))

    file.close()

    if tl_state != "e":
        print("\n******************************")
        print("Finished labelling all images!")
        print("******************************\n")
