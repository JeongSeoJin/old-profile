#!/usr/bin/env python

import rospy
from std_msgs.msg import String

# ROS 노드 초기화
rospy.init_node('error_publisher', anonymous=True)

# Publisher 설정
pub = rospy.Publisher('/error_info', String, queue_size=10)

# 메시지를 주기적으로 publish하는 함수
def publish_error_message():
    rate = rospy.Rate(0.1)  # 1Hz, 1초에 한 번 메시지 발행
    while not rospy.is_shutdown():
        error_message = "error_no_materials"
        rospy.loginfo(f"Publishing error message: {error_message}")
        pub.publish(error_message)
        rate.sleep()  # 1초 대기    

if __name__ == '__main__':
    try:
        publish_error_message()
    except rospy.ROSInterruptException:
        pass
