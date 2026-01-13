#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from std_msgs.msg import String, Bool, Int16
from macstouch.msg import order, state, vision_info, control_info, error, material

def callback(msg):
    print(f"{msg.id} done")


def main():

    rospy.init_node("gui")
    rate = rospy.Rate(10)
    order_pub = rospy.Publisher('/id_order', order, queue_size=1)
    order_sub = rospy.Subscriber('/id_complete', order, queue_size=10)
    id = 0

    while not rospy.is_shutdown():

        order_list = list(map(int, input("order : ")))
        order_info = order()

        order_info.id = id
        order_info.menu = order_list

        order_pub.publish(order_info)
        id += 1
        rate.sleep()

if __name__ == "__main__":
    main()