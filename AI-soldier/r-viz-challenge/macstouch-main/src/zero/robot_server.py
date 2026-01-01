#!/usr/bin/python
# -*- coding: utf-8 -*-

## 1．초기 설정 ①　모듈 가져오기 ######################
from i611_MCS import *
from teachdata import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import * 

from threading import Event # event 모듈
import os # 운영체제 모듈
import time # 시간관련 모듈
import math # 계산관련 모듈
import socket


MaterialList = ["bread", "meat", "cheeze", "pickle", "onion", "sauce", "tomato", "lettuce"]


def main():
    client_socket = None

    rb = i611Robot()

    _BASE = Base()

    rb.open()

    IOinit( rb )

    # gripper pin reset
    dout(48, '000')

    m = MotionParam( jnt_speed=10, lin_speed=10, pose_speed=10, overlap = 30 )

    rb.motionparam( m )


    # Set tool offset
    rb.settool(id=1, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=2, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=3, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=4, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=5, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=6, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=7, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)
    rb.settool(id=8, offx=0, offy=0, offz=200, offrz=0, offry=0, offrx=0)


    # Set motion params
    param = [

        MotionParam(jnt_speed=10, lin_speed=10, pose_speed=10, overlap = 30),
        MotionParam(jnt_speed=10, lin_speed=10, pose_speed=20, overlap = 30)

    ]


    try:

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 소켓 생성
        server_socket.bind(('192.168.1.23', 5000))  # 소켓에 IP 주소와 포트 번호 결합
        print("Server is running...")
        server_socket.listen(5)  # 클라이언트의 접속을 대기
        client_socket, addr = server_socket.accept()  # 요청 수신
        print("Connection from", addr)


        while True:
                
            Socket_data = client_socket.recv(65535) # server socket으로부터 data 수신
            if not Socket_data:
                break
            Socket_data = Socket_data.decode() # byte code -> 문자열 변환
            Motions = Socket_data.split('=')
            
            for Motion in Motions:
                motion = Motion.split('+')

                mode = motion[0]

                rb.asyncm(1)

                if mode =="MovePoint":
                    pos = [float(i) for i in motion[1].split(',')]  # 좌표로 명령 시 반드시 6개 정수
                    rb.move(Position(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]))


                elif mode =="MoveOffset":
                    offset = [float(i) for i in motion[1].split(',')]  # 좌표로 명령 시 반드시 6개 정수
                    rb.relline(dx=offset[0], dy=offset[1], dz=offset[2], drx=offset[3], dry=offset[4], drz=offset[5])


                elif mode =="MovePick":
                    # pos = [float(i) for i in motion[1].split(',')]
                    # offset = [float(i) for i in motion[2].split(',')]

                    # p1 = Position(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6])
                    # p2 = Position(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6])

                    # m_overlap = MotionParam(jnt_speed=10, lin_speed=10, pose_speed=20, overlap = 60)

                    # rb.move(p1, p2, m_overlap)

                    # rb.set_MDO()

                    client_socket.send("Done")


                elif mode =="MovePlace":

                    client_socket.send("Done")


                elif mode =="MoveSmooth":

                    client_socket.send("Done")


                elif mode =="Gripper":
                    if motion[1] == "True":
                        dout(48, '1**') # 그리퍼 열기

                    elif motion[1] == "False":
                        dout(48, '**1') # 그리퍼 닫기

                    rb.sleep(1)
                    dout(48, '000')

                    # sock.send("Done")


                elif mode =="ChangeTool":
                    id = int(motion[1])+1
                    rb.changetool(tid=id)


                elif mode =="ChangeParam":
                    rb.motionparam(param[int(motion[1])])


                elif mode =="ReadCoord":
                    position_list = shm_read( 0x3000, 6).split(',')

                    position_list[0] = round(float(position_list[0])*1000,3) # XY 좌표계 X 위치
                    position_list[1] = round(float(position_list[1])*1000,3) # XY 좌표계 Y 위치
                    position_list[2] = round(float(position_list[2])*1000,3) # XY 좌표계 Z 위치
                    position_list[3] = round(math.degrees(float(position_list[3])),3) # XY 좌표계 Rz 위치
                    position_list[4] = round(math.degrees(float(position_list[4])),3) # XY 좌표계 Ry 위치
                    position_list[5] = round(math.degrees(float(position_list[5])),3) # XY 좌표계 Rx 위치

                    print("x={0}, y={1}, z={2}, rz={3}, ry={4}, rx={5}".format(\
                        position_list[0], position_list[1], position_list[2], position_list[3], position_list[4], position_list[5]))
                    
                    cur_pos = ','.join(str(p) for p in position_list)
                    client_socket.send(cur_pos)


                elif mode =="ReadJoint":
                    joint_list = shm_read( 0x3050, 6 ).split( ',' )

                    joint_list[0] = round(math.degrees(float(joint_list[0])),3) # Joint 좌표계 J1 위치
                    joint_list[1] = round(math.degrees(float(joint_list[1])),3) # Joint 좌표계 J2 위치
                    joint_list[2] = round(math.degrees(float(joint_list[2])),3) # Joint 좌표계 J3 위치
                    joint_list[3] = round(math.degrees(float(joint_list[3])),3) # Joint 좌표계 J4 위치
                    joint_list[4] = round(math.degrees(float(joint_list[4])),3) # Joint 좌표계 J5 위치
                    joint_list[5] = round(math.degrees(float(joint_list[5])),3) # Joint 좌표계 J6 위치

                    print("j1={0}, j2={1}, j3={2}, j4={3}, j5={4}, j6={5}".format(\
                        joint_list[0], joint_list[1], joint_list[2], joint_list[3], joint_list[4], joint_list[5]))
                    
                    cur_jnt = ','.join(str(j) for j in joint_list)
                    client_socket.send(cur_jnt)


                elif mode =="ReadState":
                    pass


            client_socket.send('done')

            


    except KeyboardInterrupt:           # "ctrl" + "c" 버튼 입력
        print("KeyboardInterrupt")
    except Robot_emo:
        print("Robot_emo")
    except Exception as e:
        print("Error name is : {}".format(e))
    finally:
        print("finally")
        dout(48, '000')

    rb.asyncm(2)
    rb.close()

    client_socket.close()
    server_socket.close()
        

if __name__ == '__main__':
    main()