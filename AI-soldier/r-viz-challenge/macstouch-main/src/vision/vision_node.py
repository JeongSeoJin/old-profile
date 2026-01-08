#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from std_msgs.msg import String, Int16
from macstouch.msg import vision_info

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2
from copy import deepcopy

# from macstouch_config import MaterialList
from realsense.realsense_camera import DepthCamera

MaterialList = ["bread", "meat", "cheeze", "pickle", "onion", "sauce1", "sauce2", "tomato", "lettuce"]
VisionClass = ["pickle", "tomato"]
resolution_width, resolution_height = (1280,  720)

# model_path = '/home/choiyj/catkin_ws/src/macstouch/src/vision/pt/tomatopicklemeat.pt'
model_path = '/home/mac/catkin_ws/src/macstouch/src/vision/pt/zeus2.pt'

#!/usr/bin/env python
# -- coding: utf-8 --

# 아주 쉬운 설명 주석을 추가한 vision_node.py 입니다.
# 이 파일은 로봇이 무엇을 잡아야 하는지 "찾아서 알려주는" 역할을 해요.
# (아래 코드는 카메라로 사진을 찍고, 물건을 찾아서 잡을 좌표를 계산해서 알려줘요.)

import rospy
from std_msgs.msg import String, Int16
from macstouch.msg import vision_info

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2
from copy import deepcopy

# 카메라를 쉽게 다루기 위한 코드(다른 파일에 있어요)
from realsense.realsense_camera import DepthCamera

# 재료 목록이에요. 번호로 불러서 어떤 재료인지 알 수 있어요.
MaterialList = ["bread", "meat", "cheeze", "pickle", "onion", "sauce1", "sauce2", "tomato", "lettuce"]
VisionClass = ["pickle", "tomato"]
# 카메라 해상도 설정
resolution_width, resolution_height = (1280,  720)

# 학습된 YOLO 모델 파일 경로 (사람마다 다를 수 있어요)
model_path = '/home/mac/catkin_ws/src/macstouch/src/vision/pt/zeus2.pt'


class Vision:
    """
    Vision 클래스는 카메라에서 사진을 받고,
    물건을 찾고, 잡기 위한 좌표를 계산해서 다른 곳에 알려주는 역할을 해요.
    """
    def __init__(self) -> None:
        # 어떤 재료를 찾으라는 요청을 기다리는 곳이에요.
        self.vision_sub = rospy.Subscriber('/vision_req', Int16, self.vision_callback, queue_size=1)
        # 찾은 좌표를 알려주는 곳이에요.
        self.vision_pub = rospy.Publisher('/pick_coord', vision_info, queue_size=1)

        # 물건을 찾는 똑똑한 모델을 불러와요 (YOLO)
        self.model = YOLO(model_path)

        # 카메라 준비(리얼센스)
        self.rs = DepthCamera(resolution_width, resolution_height)
        ret, depth_raw_frame, color_raw_frame = self.rs.get_raw_frame()

        # 카메라에서 받은 화면을 저장해요.
        self.color_frame = np.asanyarray(color_raw_frame.get_data())
        self.depth_raw_frame = depth_raw_frame
        self.depth_frame = depth_raw_frame.as_depth_frame()
        self.depth_image = np.asanyarray(depth_raw_frame.get_data())
        # 모델에 한 번 이미지를 넣어 초기화해요.
        self.model(self.color_frame)

        # 깊이(거리)를 실제 단위로 바꾸는 숫자
        self.depth_scale = self.rs.get_depth_scale()

        # 화면을 저장할 변수들
        self.yolo_color = np.asanyarray(color_raw_frame.get_data())
        self.yolo_depth = np.asanyarray(depth_raw_frame.get_data())

        # 로봇의 팔이 닿을 수 있는 범위(간단한 상자 형태의 범위)
        self.coord_limit =  {"pickle":  [[-20, 20],[-20, 20], [25, 35]], 
                             "tomato":  [[-20, 20],[-20, 20], [25, 35]],
                             "lettuce": [[0, 4],[0, 0], [25, 35]], 
                             "onion":   [[0, 4],[0, 0], [25, 35]]}
        # 바운딩 박스 중심에서 몇 군데를 후보로 잡을지 정해놓았어요.
        # 각 후보는 (x 오프셋, y 오프셋, 회전 각도)
        self.pos_offset =   {"pickle":  [[0.2, -0.2, 45],[-0.2, -0.2, 45],[-0.2, 0.2, 45],[0.2, 0.2, 45]],
                             "tomato":  [[0.35, -0.35, 45],[-0.35, -0.35, 45],[-0.35, 0.35, 45],[0.35, 0.35, 45]]}
        # 화면 안에서 후보의 유효한 범위를 숫자로 정해요 (픽셀 기준)
        self.w_limit =      {"pickle":  [510, 930],
                             "tomato":  [510, 930]}
        self.h_limit =      {"pickle":  [120, 720],
                             "tomato":  [120, 720]}

        # 상자 안에서 여러 구역을 검사할 때 쓸 좌표(lettuce, onion용)
        self.rois =         {"lettuce": [[560, 100], [560, 144+100], [560, 288+100], [560, 432+100]],
                             "onion":   [[560, 100], [560, 144+100], [560, 288+100], [560, 432+100]]}
        # ROI 크기 (픽셀)
        self.wh_offset =    {"lettuce": [int(1280/3-100), int(720/5)],
                             "onion":   [int(1280/3-100), int(720/5)]}
        
        # 후보 인덱스에 따라 미리 정해둔 회전값(간단한 설정)
        self.rotation = [[-45, 0, 180], [45, 0, 180], [135, 0, 180], [-135, 0, 180]]

        # 케이스 판정할 때 쓸 작은 박스들
        self.case_roi = [(615, 70, 200, 200), (615, 420, 200, 200), (620, 340, 200, 20)]

    def vision_callback(self, msg):
        # 누군가가 어떤 재료를 찾으라고 하면 이 함수가 불러져요.
        target_idx = msg.data
        target = MaterialList[target_idx]
        print("target : ", target)

        # 실제로 찾는 일을 하는 함수에 넘겨서 결과를 받아요.
        mode, grip_pos, size = self.detection(target)

        # 결과를 로봇에게 알려줘요.
        self.pub(target_idx, mode, grip_pos, size)

    def detection(self, target):
        # 어떤 재료인지에 따라 다른 방법으로 물건을 찾게 해요.
        # 결과는 (모드, 좌표, 크기)예요.
        mode = 0
        coord = np.zeros(6)
        size = 0

        valid = False

        # 토마토와 피클은 카메라로 먼저 물건을 찾고(YOLO),
        # 그 다음 후보 지점 중에서 제일 좋은 곳을 골라요.
        if target == 'tomato':
            while valid is False:
                detected, centers, center_xy, bbox, coord = self.yolo_detection()
                print('yolo done')
                if detected:
                    mode, coord, size = self.grip_detection(target, centers, center_xy, bbox, coord)
                    print('grip done')
                    valid = self.coord_check(target, coord)
        elif target == 'pickle':
            while valid is False:
                detected, centers, center_xy, bbox, coord = self.yolo_detection()
                print('yolo done')
                if detected:
                    mode, coord, size = self.grip_detection(target, centers, center_xy, bbox, coord)
                    print('grip done')
                    valid = self.coord_check(target, coord)
                    print('valid ', valid)
        # 상추와 양파는 미리 정한 네 구역의 평균 깊이를 보고 가까운 곳을 고릅니다.
        elif target == 'lettuce':
            while valid is False:
                mode, coord = self.depth_detection(target)
                print('depth done')
                valid = self.coord_check(target, coord)
        elif target == 'onion':
            while valid is False:
                mode, coord = self.depth_detection(target)
                print('depth done')
                valid = self.coord_check(target, coord)
        # 케이스(뚜껑 같은 것)는 색과 가장자리 정보를 이용해서 왼쪽/오른쪽을 판단해요.
        elif target == 'case':
            mode, coord = self.case_detection()
            print('depth done')

        return mode, coord, size

    def coord_check(self, target, coord):
        # 계산한 좌표가 로봇 팔이 닿을 수 있는 범위 안인지 확인해요.
        x_check = self.coord_limit[target][0][0] <= coord[0] <= self.coord_limit[target][0][1]
        y_check = self.coord_limit[target][1][0] <= coord[1] <= self.coord_limit[target][1][1]
        z_check = self.coord_limit[target][2][0] <= coord[2] <= self.coord_limit[target][2][1]
        return x_check * y_check * z_check

    def yolo_detection(self):
        # 카메라 화면에서 YOLO 모델로 물건을 찾는 부분이에요.
        # 찾으면 중심 좌표(center_xy)와 박스(bbox), 그리고 실제 거리(역투영한 좌표)를 반환해요.
        results = self.model(self.color_frame)

        annotated_frame = self.color_frame.copy()
        color = [0, 255, 0]

        min_dis = 99999999999
        center_xy = [424,.240]
        centers = []

        _bbox = None

        # 중심을 고를 때 깊이(거리)를 더 중요하게 보도록 가중치를 줍니다
        center_weight = 0.0
        z_weight = 1

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf
                # 신뢰도가 0.5보다 큰 것만 사용해요.
                if confidence > 0.5:
                    xyxy = box.xyxy.tolist()[0]
                    cx = int((xyxy[2]+xyxy[0])//2)
                    cy = int((xyxy[3]+xyxy[1])//2)
                    centers.append([cx, cy])
                    
                    x, y, x2, y2 = list(map(int, xyxy)) 
                    cv2.rectangle(annotated_frame, (x, y), (x2, y2), color, 2)
                    
                    # 중심에서 얼마나 가까운지와 깊이를 합쳐서 가장 좋은 박스를 고릅니다.
                    center_dis = (cx-resolution_width/2)**2+(cy-resolution_height/2)**2
                    z_dis = round((self.depth_frame.get_distance(cx, cy) * 100), 2)
                    dis = center_weight * center_dis + z_weight * z_dis

                    if dis < min_dis:
                        min_dis = dis
                        center_xy = [cx, cy]
                        _bbox = xyxy

        if _bbox is None:
            # 아무것도 못 찾았으면 False를 돌려줘요.
            return False, [[0,0]], [0,0], [0,0,0,0], [0,0,0]
        
        # 중심 좌표는 centers 목록에서 제거해서 충돌 계산에서 제외해요.
        centers.remove(center_xy)

        color = [255, 0, 0]
        bbox = list(map(int, _bbox)) 
        x, y, x2, y2 = bbox

        # 선택한 픽셀의 깊이를 불러서, 카메라 좌표로 변환해요.
        depth = round((self.depth_frame.get_distance(center_xy[0], center_xy[1]) * 100), 2)
        wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(self.rs.depth_intrinsics, [center_xy[0], center_xy[1]], depth)
        # 화면 크기가 달라서 간단히 보정해주는 부분이에요.
        wx = round(wx*(848/1280), 3)
        wy = round(wy*(480/720), 3)
        wz = round(wz, 3)
        
        cv2.rectangle(annotated_frame, (x, y), (x2, y2), color, 2)
        cv2.line(annotated_frame, (640, 0), (640, 720), (0, 0, 255), 2)
        cv2.line(annotated_frame, (0, 360), (1280, 360), (0, 0, 255), 2)

        self.yolo_color = annotated_frame
        self.yolo_depth = np.asanyarray(self.depth_raw_frame.get_data())
        self.yolo_depth_frame = self.depth_frame

        # 찾았다는 표시와 함께 좌표를 반환해요.
        return True, centers, center_xy, bbox, [wx, wy, wz]
    
    def cost_function(self, pos, centers, h_limit, w_limit):
        # 후보 위치의 '비용'을 계산해서 좋은 위치를 골라요.
        # 다른 물체와 가까우면 안 좋고, 화면 경계에 가까우면 안 좋아요.
        if len(centers) > 0:
            obs_cost = min([np.linalg.norm(np.array(pos) - np.array(center), ord=2) for center in centers])
        else:
            obs_cost = 0
        w_cost = min(abs(pos[0] - w_limit[0]), abs(w_limit[1] - pos[0]))
        h_cost = min(abs(pos[1] - h_limit[0]), abs(h_limit[1] - pos[1]))
        wall_cost = min(h_cost, w_cost)
        print(obs_cost, wall_cost)

        # 더 작은 값이 좋은 값이에요(여기서는 단순 합산)
        return 1.0 * obs_cost + 5.0 * wall_cost
    
    def grip_detection(self, target, centers, center_xy, bbox, coord):
        # 찾은 물체 주변에 몇 군데 후보 위치를 만들고,
        # 비용을 계산해서 가장 좋은 후보를 선택해요.
        annotated_frame = self.yolo_color.copy()

        x, y, x2, y2 = bbox 
        w = x2 - x
        h = y2 - y

        offset = self.pos_offset[target]
        # 후보 좌표를 바운딩 박스 중심으로부터 계산해요.
        candidate_pos = [[center_xy[0] + x[0] * w * np.cos(x[2] * np.pi/180),
                          center_xy[1] + x[1] * h * np.cos(x[2] * np.pi/180)] for x in offset]
        
        max_cost = 0
        selected_pos = candidate_pos[0]
        selected_idx = 0
        for idx, pos in enumerate(candidate_pos):
            cost = self.cost_function(pos, centers, self.h_limit[target], self.w_limit[target])
            print(pos, cost)
            # 비용이 더 크면(여기선 더 안전한 자리라고 판단) 선택
            if cost > max_cost:
                max_cost = cost
                selected_pos = pos
                selected_idx = idx
            cv2.circle(annotated_frame, (int(pos[0]),int(pos[1])), 10, (0, 255, 0), -1, lineType=None, shift=None)
        
        selected_pos = list(map(int, selected_pos))

        # 선택한 픽셀에서 깊이를 읽어 실제 좌표로 바꿔요.
        depth = round((self.depth_frame.get_distance(selected_pos[0], selected_pos[1]) * 100), 2)
        wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(self.rs.depth_intrinsics, [selected_pos[0], selected_pos[1]], depth)
        wx = round(wx*(848/1280), 3)
        wy = round(wy*(480/720), 3)
        wz = round(wz, 3)

        color = [255, 0, 0]
        cv2.circle(annotated_frame, (selected_pos[0], selected_pos[1]), 10, color, -1, lineType=None, shift=None)
        cv2.putText(annotated_frame, "{}, {}, {}".format(wx, wy, wz), (x + 5, y + 60), 0, 1.0, color, 2)
        self.yolo_color = annotated_frame

        # 후보 인덱스에 해당하는 회전값을 골라서 같이 반환해요.
        rz = self.rotation[selected_idx][0]
        ry = self.rotation[selected_idx][1]
        rx = self.rotation[selected_idx][2]
        
        return str(selected_idx), [wx, wy, wz, rz, ry, rx], w/2
    
    def depth_detection(self, target):
        # 상추나 양파처럼 바트(통) 안에서 가장 가까운 구역을 찾는 방법이에요.
        annotated_color = self.color_frame.copy()
        annotated_depth = self.depth_image.copy()

        rois = self.rois[target]
        
        min_depth = float('inf')
        min_idx = 2

        for idx, roi in enumerate(rois):
            x, y = roi
            w, h = self.wh_offset[target]
            roi_depth = self.depth_image[y:y+h, x:x+w]

            # ROI 안에서 0보다 큰 값을 유효한 깊이로 봐요.
            valid_depth = roi_depth[roi_depth > 0]

            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
            else:
                print("ROI 내 유효한 뎁스 값이 없습니다.")
                continue
            
            cv2.rectangle(annotated_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(annotated_color, "{}".format(mean_depth), (int(x + 30), int(y + h/2)), 0, 1.0, (255, 0, 0), 2)
            print(mean_depth)
            if mean_depth < min_depth:
                min_idx = idx
                min_depth = mean_depth
        
        x, y = rois[min_idx]
        w, h = self.wh_offset[target]
        
        cv2.rectangle(annotated_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.yolo_color = annotated_color
        self.yolo_depth = annotated_depth

        # 가장 가까운 구역의 평균 깊이를 이용해 좌표를 보낸다 (간단한 형식)
        return str(min_idx), [0, 0, min_depth*0.1]
    
    def case_detection(self):
        # 상자(케이스) 확인은 색공간을 바꿔서 가장자리를 찾아서 왼쪽/오른쪽을 비교해요.
        annotated_color = self.color_frame.copy()
        annotated_depth = self.depth_image.copy()

        lab = cv2.cvtColor(annotated_color, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        contdst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(contdst, (3, 3), 0)
        canny = cv2.Canny(blurred, 50, 100)
        cv2.imshow('Canny', canny)

        roi1 = self.case_roi[0]
        roi2 = self.case_roi[1]

        cv2.rectangle(annotated_color, (roi1[0], roi1[1]), (roi1[0]+roi1[2], roi1[1]+roi1[3]), (255, 0, 0), 2)
        cv2.rectangle(annotated_color, (roi2[0], roi2[1]), (roi2[0]+roi2[2], roi2[1]+roi2[3]), (255, 0, 0), 2)

        points_in_roi1 = 0
        points_in_roi2 = 0

        x, y, w, h = roi1
        canny_roi1 = canny[y:y+h, x:x+w]
        points_in_roi1 = np.sum(canny_roi1 == 255)

        x, y, w, h = roi2
        canny_roi2 = canny[y:y+h, x:x+w]
        points_in_roi2 = np.sum(canny_roi2 == 255)

        x, y, w, h = self.case_roi[2]
        roi_depth = annotated_depth[y:y+h, x:x+w]
        mean_depth = 0

        valid_depth = roi_depth[roi_depth > 0]

        if len(valid_depth) > 0:
            mean_depth = np.mean(valid_depth)
        else:
            print("ROI 내 유효한 뎁스 값이 없습니다.")

        cv2.rectangle(annotated_color, (x, y), (x+w, y+h), (150, 150, 0), 2)
        cv2.putText(annotated_color, "{}".format(points_in_roi1), (int(roi1[0] + roi1[2]/2), int(roi1[1] + roi1[3]/2)), 0, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated_color, "{}".format(points_in_roi2), (int(roi2[0] + roi2[2]/2), int(roi2[1] + roi2[3]/2)), 0, 1.0, (0, 255, 255), 2)

        self.yolo_color = annotated_color
        self.yolo_depth = annotated_depth

        # 더 많은 선이 있는 쪽이 물건이 더 많다고 보고 왼쪽/오른쪽을 판단해요.
        return 'left' if points_in_roi1 > points_in_roi2 else 'right', [0, 0, mean_depth, 0, 0, 0]
    
    def pub(self, target, mode, grip_pos, size):
        # 결과를 vision_info 메시지로 만들어서 발행해요.
        data = vision_info()
        data.material = target
        data.grip_mode = mode
        data.coord = grip_pos
        data.size = size
        self.vision_pub.publish(data)


def main():
    # ROS 노드를 시작하고, 카메라 화면을 계속 보여주는 역할을 해요.
    rospy.init_node("vision_node")
    rate = rospy.Rate(10)
    vision = Vision()
    while not rospy.is_shutdown():
        ret, depth_raw_frame, color_raw_frame = vision.rs.get_raw_frame()

        if not color_raw_frame or not depth_raw_frame:
            continue
        
        # 최신 프레임으로 내용을 업데이트해요.
        vision.color_frame = np.asanyarray(color_raw_frame.get_data())
        vision.depth_raw_frame = depth_raw_frame
        vision.depth_frame = depth_raw_frame.as_depth_frame()
        vision.depth_image = np.asanyarray(depth_raw_frame.get_data())

        # 화면을 보여주기 위한 컬러맵 처리(깊이를 보기 쉽게 색으로 바꿔요)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(vision.depth_image, alpha=0.15), cv2.COLORMAP_JET)
        yolo_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(vision.yolo_depth, alpha=0.15), cv2.COLORMAP_JET)
        
        origin_images = np.vstack((vision.color_frame, depth_colormap))
        yolo_images = np.vstack((vision.yolo_color, yolo_depth_colormap))

        images = np.hstack((origin_images, yolo_images))
        images = cv2.resize(images, (848*2, 480*2))

        cv2.imshow('Camera and Yolo Detection', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()

    vision.rs.release()

if __name__ == "__main__":
    main()