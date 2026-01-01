#!/usr/bin/python
# -*- coding: utf-8 -*-

## 1. 초기 설정① #######################################
# 라이브러리 가져오기
from i611_MCS import *
from i611_extend import *
from i611_io import *
from teachdata import *
from rbsys import *
from i611_common import *
from i611shm import * 

def main():
  
    ## 2. 초기 설정② ####################################
    # i611 로봇 생성자
    rb = i611Robot()
    # 좌표계의 정의
    _BASE = Base()
    # 로봇과 연결 시작 초기화
    rb.open()
    # I/O 입출력 기능의 초기화 (I/O 미사용시 생략 가능 )
    IOinit( rb )
    dout(48, '000') # 그리퍼 열기
    # dout(48, '1**') # 그리퍼 열기
    # dout(48, '0**') # 그리퍼 열기
  
    ## 3. 교시 포인트 설정 ######################
    offset = 200
    tool_offset = 50
    up_offset = 60
    back_offset = 100
    p1 = Position(643,100.5,20,0,0,180)

    p1_offset = Position(643,87.5,50+offset,0,0,180)

    p_vision_tool = Position(609.056+150, 88.034, 487.483, -90.196, 0.089, -90.076)

    p_vision_offset = p_vision_tool.offset(dx = -tool_offset)
    p_vision_offset_up = p_vision_tool.offset(dz = up_offset)
    p_vision_offset_back = p_vision_offset_up.offset(dx = -back_offset)

    rb.settool(id=1, offx=0, offy=0, offz=150, offrz=0, offry=0, offrx=0) # 기본(손가락)
    rb.settool(id=2, offx=0, offy=0, offz=212, offrz=0, offry=0, offrx=0) # 양파, 양상추
    rb.settool(id=3, offx=0, offy=0, offz=225, offrz=0, offry=0, offrx=0) # bread, meat - 도윤

    rb.settool(id=4, offx=0, offy=0, offz=145, offrz=0, offry=0, offrx=0) # toamato
    
    rb.settool(id=5, offx=0, offy=0, offz=225, offrz=0, offry=0, offrx=0) # pickle
    rb.settool(id=6, offx=0, offy=0, offz=245, offrz=0, offry=0, offrx=0) # bread, meat - 유진
    rb.settool(id=7, offx=0, offy=0, offz=280, offrz=0, offry=0, offrx=0) # 양상추 ver2

    print(p1.pos2list())

    # j1 = Joint( 230, -1, -92, 90, 5, 89 )
    
    ## 4. 동작 조건 설정 ##################################
    #MotionParam 생성자에서 동작 조건 설정
    m = MotionParam( jnt_speed=5, lin_speed=10 )
    #MotionParam 형으로 동작 조건 설정
    rb.motionparam( m )
    rb.changetool(1)

    ## 5. 로봇 동작을 정의 ##############################
    # 작업 시작
    # rb.home()
    rb.move(p_vision_offset)

    rb.move(p_vision_tool)
    rb.sleep(0.5)

    
    m = MotionParam( jnt_speed=3, lin_speed=10 , overlap = 0)
    #MotionParam 형으로 동작 조건 설정
    rb.motionparam( m )
    
    # rb.asyncm(1)
    # rb.line(p_vision_offset_up)
    rb.line(p_vision_offset_up)
    #rb.relline(dz=up_offset)
    rb.sleep(0.5)
    rb.move(p_vision_offset_back)
    rb.changetool(6)
    #rb.join()

    # rb.asyncm(2)

    m = MotionParam( jnt_speed=3, lin_speed=10 )
    #MotionParam 형으로 동작 조건 설정
    rb.motionparam( m )

    # rb.move(p1_offset)
    
    # dout(48, '1**') # 그리퍼 열기
    # rb.sleep(0.1)
    # dout(48, '0**') # 그리퍼 열기

    # rb.move(p1)
    # rb.sleep(1)

    # dout(48, '**1')
    # rb.sleep(0.1)
    # dout(48, '**0')    
    
    # rb.move(p1_offset)
    # dout(48, '1**') # 그리퍼 열기
    # rb.sleep(0.1)
    # dout(48, '0**') # 그리퍼 열기    
    

    # rb.sleep(3)    
    # rb.move(p1)
    # dout(48, '1**') # 그리퍼 열기
    # rb.sleep(0.1)
    # dout(48, '0**') # 그리퍼 열기
    # rb.move(p1_offset)




    ## 6. 종료 ######################################
    # 로봇과의 연결을 종료
    rb.close()

if __name__ == '__main__':
    main()