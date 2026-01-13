#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from macstouch.msg import order  # order.msg를 생성 후 컴파일하면 자동 생성
from selected_menu import menu_index

# 불고기 버거 = [2, 1, 0, 3, 8, 1, 1, 3]
# 제로 버거 = [2, 0, 0, 3, 8, 1, 1, 3]
# 치즈 버거 = [2, 1, 1, 3, 8, 1, 1, 3]
# 더블 치즈 버거 = [2, 1, 2, 3, 8, 1, 1, 3]
i = 0

def menu_callback(data):
    global i
    rospy.loginfo(f"Received data: {data.data}")

    for_custom_list = [0, 0, 0, 0, 0, 0, 0, 0]

    menu_index = data.data

    i += 1
    
    # order 메시지 생성
    order_msg = order()
    order_msg.id = i

    for_len = []


    ############ menu index에서 데이터만 뽑아 custom_list에 재배열

    for j in range(len(menu_index)):
        if menu_index[j].isdigit():  # 숫자인 경우만 변환
            for_custom_list[int(menu_index[j])] += 1
        else:
            print(f"Invalid value: {menu_index[j]}")
    ############################

    # for_len = for_custom_list

    # if 0 in for_len:
    #     for_len.remove(0)
    # else:
    #     print("0 is not in the list")
    print(len(menu_index))

    if (len(menu_index)) > 3:
        order_msg.menu = for_custom_list[:]

    elif(len(menu_index)) == 3:
        if menu_index == "[0]":
            order_msg.menu = [0, 1, 0, 3, 8, 1, 1, 3]  # 불고기 버거
        elif menu_index == "[1]":
            order_msg.menu = [1, 0, 0, 3, 8, 1, 1, 3]  # 제로 버거
        elif menu_index == "[2]":
            order_msg.menu = [2, 1, 1, 3, 8, 1, 1, 3]  # 치즈 버거
        elif menu_index == "[3]":
            order_msg.menu = [3, 1, 2, 3, 8, 1, 1, 3]  # 더블 치즈 버거
        else:
            rospy.logwarn("Invalid menu index received!")
            return

    # 퍼블리시
    order_pub.publish(order_msg)
    rospy.loginfo(f"Published: order_id={order_msg.id}, menu_list={order_msg.menu}")

def listener_and_publisher():
    rospy.init_node('menu_order_node', anonymous=True)

    # 퍼블리셔 생성: /order 토픽에 order 메시지 퍼블리시
    global order_pub
    order_pub = rospy.Publisher('/id_order', order, queue_size=10)

    # 서브스크라이버 생성: /menu_index 토픽을 구독
    rospy.Subscriber('/menu_index', String, menu_callback)

    rospy.loginfo("Listening on /menu_index and publishing to /id_order...")
    rospy.spin()

if __name__ == '__main__':
    try:
        listener_and_publisher()
    except rospy.ROSInterruptException:
        pass
