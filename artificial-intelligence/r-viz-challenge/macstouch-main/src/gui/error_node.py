#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from tkinter import Tk, Button, Label

# ROS 초기화 및 설정
rospy.init_node('error_node', anonymous=True)
# pub = rospy.Publisher('/menu_list', String, queue_size=10)
error_complete_pub = rospy.Publisher('/error_complete', String, queue_size=10)  # error_complete 퍼블리셔 추가

def publish_error_complete():
    # "/error_complete" 토픽에 "error_complete" 메시지를 발행하는 함수
    message = True 
    rospy.loginfo(f"Publishing: {message}")
    error_complete_pub.publish(message)

def callback(msg):
    win_width = 350
    win_height = 700
    win_x = int(540 - win_width/2)
    win_y = int(960 - win_height/2)
    window = Tk()
    window.title("Menu Publish & Error")
    window.geometry(f"{win_height}x{win_width}+{win_x}+{win_y}")  # 창 크기 설정

    # "Error" 텍스트를 출력할 라벨 생성 (버튼 위에 위치)
    error_label = Label(window, text=msg.data, fg="black", font=("Arial", 20))
    error_label.pack(padx=20, pady=80)  # 텍스트를 위쪽에 배치

    # 버튼 생성 (텍스트 아래에 위치)
    button = Button(
        window, 
        text="확인", 
        command=lambda: [publish_error_complete(), window.destroy()],  # 버튼 클릭 시 메시지 발행 후 창 닫기
        width=10,
        height=3
    )
    button.pack(pady=20)  # 버튼을 텍스트 아래에 배치

    # Tkinter 창 실행
    window.mainloop()

# ROS 구독자 설정
sub = rospy.Subscriber('/error_info', String, callback)

# ROS 스핀을 계속해서 돌려 메시지 처리
rospy.spin()
