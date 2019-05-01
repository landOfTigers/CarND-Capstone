#!/usr/bin/env python

import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
MAX_DECELERATION = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_waypoint_idx = -1
        self.current_velocity = 0.0
        self.start_position = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg
        if self.start_position is None:
            self.start_position = self.get_closest_waypoint_idx()

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def waypoints_cb(self, msg):
        self.base_lane = msg
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 msg.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        closest_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        current_pos_vec = np.array([x, y])

        closest_is_behind = np.dot(closest_vec - prev_vec, current_pos_vec - closest_vec) > 0

        if closest_is_behind:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_lane.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        rospy.loginfo("Closest wp idx: {}".format(closest_idx))
        rospy.loginfo("Red light stop line: {}".format(self.stopline_waypoint_idx))
        rospy.loginfo("Farthest wp idx: {}".format(farthest_idx))

        initial_wp_delay = 4
        if (0 <= self.stopline_waypoint_idx <= farthest_idx) and (closest_idx > self.start_position + initial_wp_delay):
            rospy.loginfo("Decelerating...")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        else:
            rospy.loginfo("Going normally")
            lane.waypoints = base_waypoints

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        stop_idx = max(self.stopline_waypoint_idx - closest_idx - 2, 0)

        velocities = np.linspace(0.0, self.current_velocity, endpoint=False, num=(stop_idx + 1)).tolist()[::-1]
        velocities.extend([0.0] * (LOOKAHEAD_WPS - stop_idx - 1))

        for velocity, waypoint in zip(velocities, waypoints):
            wp = Waypoint()
            wp.pose = waypoint.pose
            wp.twist.twist.linear.x = min(velocity, waypoint.twist.twist.linear.x)
            temp.append(wp)

        return temp

    def traffic_cb(self, msg):
        self.stopline_waypoint_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # TODO: use these methods for better readability
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
