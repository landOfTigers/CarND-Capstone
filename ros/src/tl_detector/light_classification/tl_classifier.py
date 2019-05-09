import os

import numpy as np
import rospy
from keras.models import load_model


class TLClassifier(object):
    def __init__(self):
        model_path = '%s/model.h5' % os.path.dirname(os.path.realpath(__file__))
        self.model = load_model(model_path)
        self.model._make_predict_function()  # workaround for "Tensor is not an element of this graph" bug

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        predictions = self.model.predict(image[None, :, :, :], batch_size=1)
        rospy.loginfo('Predictions: %s' % predictions)
        classification = np.argmax(predictions[0])
        if classification == 3:
            classification = 4
        return classification
