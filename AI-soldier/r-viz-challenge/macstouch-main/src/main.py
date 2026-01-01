#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from std_msgs.msg import String, Bool, Int16, Float32MultiArray
from macstouch.msg import order, state, vision_info, control_info, error, material

import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

# from macstouch_config import *
from task_maker import Task

MaterialList = ["bread", "meat", "cheeze", "pickle", "onion", "sauce1", "sauce2", "tomato", "lettuce"]
Loadcell = ["bread", "meat", "lettuce", "tomato", "onion", "pickle"]
WeightLimit = [100, -100, 100, 100, 100, -1000]
ModeList = ["init_pos", "vision", "pnp", "tool_return"]


OrderList = []
IngId = None


class Burger:
    def __init__(self) -> None:
        self.id = None
        self.menu = None
        self.state = "NotYet" # NotYet / Ing / Done


class GUI:
    def __init__(self) -> None:
        self.order_sub = rospy.Subscriber('/id_order', order, self.order_callback, queue_size=1)
        self.order_pub = rospy.Publisher('/id_complete', order, queue_size=10)

    def order_callback(self, msg):
        global OrderList

        burger = Burger()
        burger.id = msg.id
        burger.menu = list(msg.menu)

        OrderList.append(burger)

        print(f"Order ID{msg.id} added !!")

    def order_complete(self, id):
        global OrderList
        OrderList.pop(0)

        complete_msg = order()
        complete_msg.id = id
        self.order_pub.publish(complete_msg)



class WorkSpace:
    def __init__(self):
        self.mat_sub = rospy.Subscriber('/material_info', Float32MultiArray, self.mat_callback, queue_size=10)
        self.mat_pub = rospy.Publisher('/error_info', String, queue_size=10)
        self.weights_limit = np.array(WeightLimit)
        self.weights = []

    def mat_callback(self, msg):
        self.weights = np.array(msg.data)
        self.mat_check()

    def mat_check(self):
        lack_mat = ""
        lack_str = "Lack of material: "
        dis = self.weights - self.weights_limit
        for idx, dis in enumerate(dis):
            if dis < 0:
                self.mat_pub.publish(lack_str + Loadcell[idx])

    def pickup_check(self):
        pass


class Vision:
    def __init__(self) -> None:
        self.vision_sub = rospy.Subscriber('/pick_coord', vision_info, self.vision_callback, queue_size=1)
        self.coords = [{'ready': False, 'grip_mode': 'x', 'coord': [0,0,0,0,0,0], 'size': 0}] * len(MaterialList)

    def vision_callback(self, msg):
        self.coords[msg.material] = {'ready': True, 'grip_mode': msg.grip_mode, 'coord': msg.coord, 'size': msg.size}


class Control:
    def __init__(self) -> None:
        self.state_sub = rospy.Subscriber('/robot_state', state, self.state_callback, queue_size=1)
        # self.error_pub = rospy.Publisher('/error_info', error, queue_size=10)

    def state_callback(self, msg):
        pass

    def error_check(self):
        pass


class Request:
    def __init__(self):
        self.vision = Vision()
        self.vision_pub = rospy.Publisher('/vision_req', Int16, queue_size=10)
        self.control_pub = rospy.Publisher('/control_req', control_info, queue_size=10)
        self.workspace_pub = rospy.Publisher('/workspace', String, queue_size=10)
        self.done_sub = rospy.Subscriber('/done', Bool, self.done_callback)
        
        self.task_list = []
        self.current_task = None

    def vision_req(self, material):
        rospy.sleep(0.5)
        self.vision_pub.publish(material)

    def control_req(self, mode, material=None, grip_mode=None, coord=None, size=None):
        request = control_info()
        print(mode)

        if mode == 'init_pos': # init_pos
            request.mode = mode
            request.material = -1
            request.grip_mode = '0'
            request.coord = [0, 0, 0, 0, 0, 0]
            request.size = 30

        elif mode == 'tool_get': # tool_get
            request.mode = mode
            request.material = material
            request.grip_mode = '0'
            request.coord = [0, 0, 0, 0, 0, 0]
            request.size = 0

        elif mode == 'vision': # vision
            request.mode = mode
            request.material = material
            request.grip_mode = '0'
            request.coord = [0, 0, 0, 0, 0, 0]
            request.size = 0

        elif mode == 'pnp': # pnp
            request.mode = mode
            request.material = material
            request.grip_mode = grip_mode
            request.coord = coord
            request.size = OrderList[0].menu[3]

        elif mode == 'tool_return': # tool_return
            request.mode = mode
            request.material = material
            request.grip_mode = '0'
            request.coord = [0, 0, 0, 0, 0, 0]
            request.size = 0

        elif mode == 'finish': # finish
            request.mode = mode
            request.material = 0
            request.grip_mode = '0'
            request.coord = [0, 0, 0, 0, 0, 0]
            request.size = 0

        self.control_pub.publish(request)

    def workspace_req(self, args):
        pass

    def done_callback(self, msg):
        self.current_task[1] = 1

    def task_control(self, step):
        status = 'Ing'
        if self.current_task is None:
            self.current_task = [self.task_list[step], 0]

        if self.current_task[1] == 0:
            print(self.current_task)
            if self.current_task[0]['mode'] == 'pnp':
                self.control_req(self.current_task[0]['mode'], self.current_task[0]['material'],
                                 grip_mode=self.vision.coords[self.current_task[0]['material']]['grip_mode'],
                                 coord=self.vision.coords[self.current_task[0]['material']]['coord'],
                                 size=self.vision.coords[self.current_task[0]['material']]['size'])
                self.vision.coords[self.current_task[0]['material']]['ready'] = False
            else:
                self.control_req(self.current_task[0]['mode'], self.current_task[0]['material'])
            self.current_task[1] = -1


        elif self.current_task[1] == 1:
            if self.current_task[0]['mode'] == 'vision':
                self.vision_req(self.current_task[0]['material'])
                self.current_task[1] = 2
            else:
                step += 1
                self.current_task = None

        elif self.current_task[1] == 2:
            if self.vision.coords[self.current_task[0]['material']]['ready']:
                step += 1
                self.current_task = None

        if step > len(self.task_list)-1:
            status = 'Done'

        return step, status


def main():
    global IngId, OrderList

    rospy.init_node("main_node")
    rate = rospy.Rate(10)

    req = Request()

    gui = GUI()
    workspace = WorkSpace()
    control = Control()
    task = Task()

    step = 0
    status = None

    while not rospy.is_shutdown():
        print("===========================================")
        if IngId is None and len(OrderList) != 0:
            IngId = OrderList[0].id
            OrderList[0].state = "Ing"
            req.task_list = task.order_to_task(OrderList[0].menu)
            print(req.task_list)
        
        if IngId is None:
            print('주문이 없습니다.')
            rate.sleep()
            continue

        step, status = req.task_control(step)

        if status != 'Done':
            print("들어온 주문 수 : ", len(OrderList))
            print("현재 진행 중인 주문 : ", IngId)
            print(f"현재 진행 중인 단계 : {step} {req.task_list[step]}")

        else:
            print("{} 제작이 완료되었습니다.".format(IngId))
            gui.order_complete(IngId)
            IngId = None
            step = 0
            status = None
            req.task_list = None



if __name__ == "__main__":
    main()