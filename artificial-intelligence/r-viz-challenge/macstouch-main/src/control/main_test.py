# main_test
#!/usr/bin/env python3

import rospy
import numpy as np
import math
import time
import json
import os, sys

# msg
from std_msgs.msg import Float32MultiArray as fl
from std_msgs.msg import Bool, Float32, String
from macstouch.msg import control_info
from macstouch.msg import action_info

def callback(data):
    print('Mode 동작 완료')
    
def main():
    rospy.Subscriber('/done', Bool, callback)
    rate = rospy.Rate(10) # 10hz
    test_index = 5
    while not rospy.is_shutdown():
        mode = input("Mode : ") # Mode 입력
        print('[topic data]')
        control_req = rospy.Publisher('/control_req', control_info, queue_size=10)    
        control_data = control_info()
        

        if mode== 'init_pos':
            control_data.mode = 'init_pos'
            control_data.material = -1
            control_data.grip_mode = 'x'
            control_data.coord = [0, 0, 0, 0, 0, 0]
            control_data.size = 30
            control_req.publish(control_data)
            print('Mode : init_pos')
            print(control_data)

        elif mode== 'vision':
            control_data.mode = 'vision'
            control_data.material = test_index
            control_data.grip_mode = 'x'
            control_data.coord = [0, 0, 0, 0, 0, 0]
            control_data.size = 0
            control_req.publish(control_data)
            print('Mode : vision')
            print(control_data)

        elif mode == 'pnp':
            control_data.mode = 'pnp'
            control_data.material = test_index
            control_data.grip_mode = '0'
            # control_data.coord = [500,500,300,-15,0,180]
            control_data.coord = [0, 0, 25,-90, 0,-180]            
            control_data.size = 30
            control_req.publish(control_data)
            print('Mode : pnp')
            print(control_data)

        elif mode == 'tool_return':
            control_data.mode = 'tool_return'
            control_data.material = test_index
            control_data.grip_mode = 'x'
            control_data.coord = [0, 0, 0, 0, 0, 0]
            control_data.size = 0
            control_req.publish(control_data)
            print('Mode : tool_return')
            print(control_data)

        elif mode == 'tool_get':
            control_data.mode = 'tool_get'
            control_data.material = test_index
            control_data.grip_mode = 'x'
            control_data.coord = [0, 0, 0, 0, 0, 0]
            control_data.size = 0
            control_req.publish(control_data)
            print('Mode : tool_get')
            print(control_data)   

        elif mode == 'finish':
            control_data.mode = 'finish'
            control_data.material = -1
            control_data.grip_mode = 'x'
            control_data.coord = [0, 0, 0, 0, 0, 0]
            control_data.size = 0
            control_req.publish(control_data)
            print('Mode : finish')
            print(control_data)    

        else:
            print('해당 mode는 존재하지 않습니다.')

        print('###################')
        rate.sleep()


        





if __name__ == '__main__':
    try:
        rospy.init_node('main_test', anonymous=True)
        rospy.loginfo("main_test_node is on")
        main()

    except rospy.ROSInterruptException:
        rospy.loginfo('Program is shut down')