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

class Vision:
    def __init__(self) -> None:
        self.vision_sub = rospy.Subscriber('/vision_req', Int16, self.vision_callback, queue_size=1)
        self.vision_pub = rospy.Publisher('/pick_coord', vision_info, queue_size=1)

        self.model = YOLO(model_path)

        self.rs = DepthCamera(resolution_width, resolution_height)
        ret, depth_raw_frame, color_raw_frame = self.rs.get_raw_frame()

        self.color_frame = np.asanyarray(color_raw_frame.get_data())
        self.depth_raw_frame = depth_raw_frame
        self.depth_frame = depth_raw_frame.as_depth_frame()
        self.depth_image = np.asanyarray(depth_raw_frame.get_data())
        self.model(self.color_frame)

        self.depth_scale = self.rs.get_depth_scale()

        self.yolo_color = np.asanyarray(color_raw_frame.get_data())
        self.yolo_depth = np.asanyarray(depth_raw_frame.get_data())

        # 그리퍼가 도달 가능한 범위 내에 있는지 확인하기 위한 좌표 (월드 좌표계)
        self.coord_limit =  {"pickle":  [[-20, 20],[-20, 20], [25, 35]], 
                             "tomato":  [[-20, 20],[-20, 20], [25, 35]],
                             "lettuce": [[0, 4],[0, 0], [25, 35]], 
                             "onion":   [[0, 4],[0, 0], [25, 35]]}
        # 후보 grip coord를 생성하기 위한 값 ( x, y, theta )
        self.pos_offset =   {"pickle":  [[0.2, -0.2, 45],[-0.2, -0.2, 45],[-0.2, 0.2, 45],[0.2, 0.2, 45]],
                             "tomato":  [[0.35, -0.35, 45],[-0.35, -0.35, 45],[-0.35, 0.35, 45],[0.35, 0.35, 45]]}
        # cost 계산을 위한 바트 높이 (픽셀 좌표계)
        self.w_limit =      {"pickle":  [510, 930],
                             "tomato":  [510, 930]}
        # cost 계산을 위한 바트 너비 (픽셀 좌표계)
        self.h_limit =      {"pickle":  [120, 720],
                             "tomato":  [120, 720]}

        # 바트 안에 4개 구역 ROI 시작점 설정 (픽셀 좌표계)
        self.rois =         {"lettuce": [[560, 100], [560, 144+100], [560, 288+100], [560, 432+100]],
                             "onion":   [[560, 100], [560, 144+100], [560, 288+100], [560, 432+100]]}
        # ROI 너비, 높이 설정 (픽셀 좌표계)
        self.wh_offset =    {"lettuce": [int(1280/3-100), int(720/5)],
                             "onion":   [int(1280/3-100), int(720/5)]}
        
        self.rotation = [[-45, 0, 180], [45, 0, 180], [135, 0, 180], [-135, 0, 180]]

        self.case_roi = [(615, 70, 200, 200), (615, 420, 200, 200), (620, 340, 200, 20)]

    def vision_callback(self, msg):
        target_idx = msg.data
        target = MaterialList[target_idx]
        print("target : ", target)

        mode, grip_pos, size = self.detection(target)

        self.pub(target_idx, mode, grip_pos, size)

    def detection(self, target):
        mode = 0
        coord = np.zeros(6)
        size = 0

        valid = False

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
        elif target == 'case':
            mode, coord = self.case_detection()
            print('depth done')
        # else:
        #     while valid is False:
        #         mode, coord = 0, [coord[0], coord[1], coord[2], 0, 0, 0]
        #         valid = self.coord_check(target, coord)

        return mode, coord, size

    def coord_check(self, target, coord):
        x_check = self.coord_limit[target][0][0] <= coord[0] <= self.coord_limit[target][0][1]
        y_check = self.coord_limit[target][1][0] <= coord[1] <= self.coord_limit[target][1][1]
        z_check = self.coord_limit[target][2][0] <= coord[2] <= self.coord_limit[target][2][1]
        return x_check * y_check * z_check

    def yolo_detection(self):
        results = self.model(self.color_frame)

        annotated_frame = self.color_frame.copy()
        color = [0, 255, 0]

        min_dis = 99999999999
        center_xy = [424,240]
        centers = []

        _bbox = None

        center_weight = 0.0
        z_weight = 1

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf
                if confidence > 0.5:
                    xyxy = box.xyxy.tolist()[0]
                    cx = int((xyxy[2]+xyxy[0])//2)
                    cy = int((xyxy[3]+xyxy[1])//2)
                    centers.append([cx, cy])
                    
                    x, y, x2, y2 = list(map(int, xyxy)) 
                    cv2.rectangle(annotated_frame, (x, y), (x2, y2), color, 2)
                    
                    center_dis = (cx-resolution_width/2)**2+(cy-resolution_height/2)**2
                    z_dis = round((self.depth_frame.get_distance(cx, cy) * 100), 2)
                    dis = center_weight * center_dis + z_weight * z_dis

                    if dis < min_dis:
                        min_dis = dis
                        center_xy = [cx, cy]
                        _bbox = xyxy

        if _bbox is None:
            return False, [[0,0]], [0,0], [0,0,0,0], [0,0,0]
        
        centers.remove(center_xy)

        color = [255, 0, 0]
        bbox = list(map(int, _bbox)) 
        x, y, x2, y2 = bbox

        depth = round((self.depth_frame.get_distance(center_xy[0], center_xy[1]) * 100), 2)
        wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(self.rs.depth_intrinsics, [center_xy[0], center_xy[1]], depth)
        wx = round(wx*(848/1280), 3)
        wy = round(wy*(480/720), 3)
        wz = round(wz, 3)
        
        # cv2.putText(annotated_frame, "{}, {}, {}".format(wx, wy, wz), (x + 5, y + 60), 0, 1.0, color, 2)

        cv2.rectangle(annotated_frame, (x, y), (x2, y2), color, 2)
        cv2.line(annotated_frame, (640, 0), (640, 720), (0, 0, 255), 2)
        cv2.line(annotated_frame, (0, 360), (1280, 360), (0, 0, 255), 2)

        self.yolo_color = annotated_frame
        self.yolo_depth = np.asanyarray(self.depth_raw_frame.get_data())
        self.yolo_depth_frame = self.depth_frame

        return True, centers, center_xy, bbox, [wx, wy, wz]
    
    def cost_function(self, pos, centers, h_limit, w_limit):
        if len(centers) > 0:
            obs_cost = min([np.linalg.norm(np.array(pos) - np.array(center), ord=2) for center in centers])
        else:
            obs_cost = 0
        w_cost = min(abs(pos[0] - w_limit[0]), abs(w_limit[1] - pos[0]))
        h_cost = min(abs(pos[1] - h_limit[0]), abs(h_limit[1] - pos[1]))
        wall_cost = min(h_cost, w_cost)
        print(obs_cost, wall_cost)

        return 1.0 * obs_cost + 5.0 * wall_cost # if wall_cost <= 50 else 1.0 * obs_cost + wall_cost*0.000001
    
    def grip_detection(self, target, centers, center_xy, bbox, coord):
        annotated_frame = self.yolo_color.copy()

        x, y, x2, y2 = bbox
        w = x2 - x
        h = y2 - y

        offset = self.pos_offset[target]
        candidate_pos = [[center_xy[0] + x[0] * w * np.cos(x[2] * np.pi/180),
                          center_xy[1] + x[1] * h * np.cos(x[2] * np.pi/180)] for x in offset]
        
        max_cost = 0
        selected_pos = candidate_pos[0]
        selected_idx = 0
        for idx, pos in enumerate(candidate_pos):
            cost = self.cost_function(pos, centers, self.h_limit[target], self.w_limit[target])
            print(pos, cost)
            if cost > max_cost:
                max_cost = cost
                selected_pos = pos
                selected_idx = idx
            cv2.circle(annotated_frame, (int(pos[0]),int(pos[1])), 10, (0, 255, 0), -1, lineType=None, shift=None)
        
        selected_pos = list(map(int, selected_pos))

        depth = round((self.depth_frame.get_distance(selected_pos[0], selected_pos[1]) * 100), 2)
        wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(self.rs.depth_intrinsics, [selected_pos[0], selected_pos[1]], depth)
        wx = round(wx*(848/1280), 3)
        wy = round(wy*(480/720), 3)
        wz = round(wz, 3)

        color = [255, 0, 0]
        cv2.circle(annotated_frame, (selected_pos[0], selected_pos[1]), 10, color, -1, lineType=None, shift=None)
        cv2.putText(annotated_frame, "{}, {}, {}".format(wx, wy, wz), (x + 5, y + 60), 0, 1.0, color, 2)
        self.yolo_color = annotated_frame

        rz = self.rotation[selected_idx][0]
        ry = self.rotation[selected_idx][1]
        rx = self.rotation[selected_idx][2]
        
        return str(selected_idx), [wx, wy, wz, rz, ry, rx], w/2
    
    def depth_detection(self, target):
        annotated_color = self.color_frame.copy()
        annotated_depth = self.depth_image.copy()

        rois = self.rois[target]
        
        min_depth = float('inf')
        min_idx = 2

        for idx, roi in enumerate(rois):
            x, y = roi
            w, h = self.wh_offset[target]
            roi_depth = self.depth_image[y:y+h, x:x+w]

            # ROI 내 유효한 뎁스 값 필터링
            valid_depth = roi_depth[roi_depth > 0]  # 유효한 뎁스 값만 사용 (0은 무효한 값)

            # ROI의 평균 뎁스 계산
            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
            else:
                print("ROI 내 유효한 뎁스 값이 없습니다.")
                continue
            
            cv2.rectangle(annotated_color, (x, y), (x+w, y+h), (255, 0, 0), 2)  # ROI 사각형 그리기
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
        # wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(self.rs.depth_intrinsics, (int(rois[min_idx][0]+w/2), int(rois[min_idx][1]+h/2)), min_depth)

        return str(min_idx), [0, 0, min_depth*0.1]
    
    def case_detection(self):
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

        roi1 = self.case_roi[0]  # (x, y, width, height)
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

        # ROI 내 유효한 뎁스 값 필터링
        valid_depth = roi_depth[roi_depth > 0]  # 유효한 뎁스 값만 사용 (0은 무효한 값)

        # ROI의 평균 뎁스 계산
        if len(valid_depth) > 0:
            mean_depth = np.mean(valid_depth)
        else:
            print("ROI 내 유효한 뎁스 값이 없습니다.")

        cv2.rectangle(annotated_color, (x, y), (x+w, y+h), (150, 150, 0), 2)  # ROI 사각형 그리기
        cv2.putText(annotated_color, "{}".format(points_in_roi1), (int(roi1[0] + roi1[2]/2), int(roi1[1] + roi1[3]/2)), 0, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated_color, "{}".format(points_in_roi2), (int(roi2[0] + roi2[2]/2), int(roi2[1] + roi2[3]/2)), 0, 1.0, (0, 255, 255), 2)

        self.yolo_color = annotated_color
        self.yolo_depth = annotated_depth


        return 'left' if points_in_roi1 > points_in_roi2 else 'right', [0, 0, mean_depth, 0, 0, 0]

    
    def pub(self, target, mode, grip_pos, size):
        data = vision_info()
        data.material = target
        data.grip_mode = mode
        data.coord = grip_pos
        data.size = size
        self.vision_pub.publish(data)


def main():
    rospy.init_node("vision_node")
    rate = rospy.Rate(10)
    vision = Vision()
    while not rospy.is_shutdown():
        ret, depth_raw_frame, color_raw_frame = vision.rs.get_raw_frame()

        if not color_raw_frame or not depth_raw_frame:
            continue
        
        vision.color_frame = np.asanyarray(color_raw_frame.get_data())
        vision.depth_raw_frame = depth_raw_frame
        vision.depth_frame = depth_raw_frame.as_depth_frame()
        vision.depth_image = np.asanyarray(depth_raw_frame.get_data())

        # vision.detection('case')
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