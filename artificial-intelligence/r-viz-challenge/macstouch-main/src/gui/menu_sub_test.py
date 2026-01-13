#!/usr/bin/env python

import rospy
from std_msgs.msg import String
# from selected_menu import menu2pub  # 외부 파일의 selected 리스트

# ROS 초기화 및 설정
rospy.init_node('tkinter_ros_node', anonymous=True)

# 콜백 함수: 메시지를 수신할 때 호출됩니다.
def callback(msg):
    # 수신된 메시지를 출력
    rospy.loginfo(f"Received: {msg.data}")

# ROS 구독자 설정: '/error' 토픽을 구독
sub = rospy.Subscriber('/id_order', String, callback)

# ROS 스핀을 계속해서 돌려 메시지 처리
rospy.spin()
##sdasd   