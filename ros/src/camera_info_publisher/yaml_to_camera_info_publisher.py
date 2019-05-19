#!/usr/bin/env python

"""
Thanks to Github user @rossbar for writing this script
https://gist.github.com/rossbar/ebb282c3b73c41c1404123de6cea4771

pointgrey_camera_driver (at least the version installed with apt-get) doesn't
properly handle camera info in indigo.
This node is a work-around that will read in a camera calibration .yaml
file (as created by the cameracalibrator.py in the camera_calibration pkg),
convert it to a valid sensor_msgs/CameraInfo message, and publish it on a
topic.

The yaml parsing is courtesy ROS-user Stephan:
    http://answers.ros.org/question/33929/camera-calibration-parser-in-python/

This file just extends that parser into a rosnode.
"""
import yaml

import rospy
from sensor_msgs.msg import CameraInfo


def yaml_to_camera_info(calibration_yaml):
    """
    Parse a yaml file containing camera calibration data (as produced by
    rosrun camera_calibration cameracalibrator.py) into a
    sensor_msgs/CameraInfo msg.

    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data

    Returns
    -------
    msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    calib_data = yaml.load(calibration_yaml)
    # Parse
    msg = CameraInfo()
    msg.width = calib_data["image_width"]
    msg.height = calib_data["image_height"]
    msg.K = calib_data["camera_matrix"]["data"]
    msg.D = calib_data["distortion_coefficients"]["data"]
    msg.R = calib_data["rectification_matrix"]["data"]
    msg.P = calib_data["projection_matrix"]["data"]
    msg.distortion_model = calib_data["distortion_model"]
    return msg


if __name__ == "__main__":

    calib_yaml = rospy.get_param("/grasshopper_calibration_yaml")

    # Parse yaml file
    camera_info_msg = yaml_to_camera_info(calib_yaml)

    # Initialize publisher node
    rospy.init_node("camera_info_publisher", anonymous=True)
    publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=10)
    rate = rospy.Rate(10)

    # Run publisher
    while not rospy.is_shutdown():
        publisher.publish(camera_info_msg)
        rate.sleep()
