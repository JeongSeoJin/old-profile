import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
from ultralytics import YOLO
import pyrealsense2
import numpy as np
from math import *
from realsense.realsense_camera import DepthCamera

CLASSES = ["bread", "meat", "cheeze", "pickle", "onion", "sauce", "tomato", "cabage"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3)) # RGB
resolution_width, resolution_height = (640, 480)

model = YOLO('./pt/tomatopicklemeat.pt')

# cap = cv2.VideoCapture(2)

# if not cap.isOpened():
#     print("cannot open the device")
#     exit()
# rs = DepthCamera(resolution_width, resolution_height)
# depth_scale = rs.get_depth_scale()


while True:
#     ret, depth_raw_frame, color_raw_frame = rs.get_raw_frame()
#     color_frame = np.asanyarray(color_raw_frame.get_data())
#     depth_frame = np.asanyarray(depth_raw_frame.get_data())

    color_frame = cv2.imread("/home/choiyj/Downloads/rviz_dataset/tomato/color_0636.png")
    depth_frame = cv2.imread("/home/choiyj/Downloads/rviz_dataset/tomato/depth_0636.png", cv2.IMREAD_ANYDEPTH)

    # frame = cv2.imread("./data/image.png")
    # ret, frame = cap.read()
    # if not ret:
    #     print("cannot open the frame")
    #     break

    results = model(color_frame)

    annotated_frame = color_frame.copy()
    # depth_info = depth_raw_frame.as_depth_frame()

    # for result in results[0].boxes:
    #     if result.conf >= 0.7: 
    #         annotated_frame = results[0].plot()


    class_ids = []
    confidences = []
    bboxes = []
    obj_centers = []
    min_dis = 99999999999
    center_idx = -1
    center_xy = None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf
            if confidence > 0.0:
                xyxy = box.xyxy.tolist()[0]
                bboxes.append(xyxy)
                confidences.append(float(confidence))
                class_ids.append(box.cls.tolist())
                cx = int((xyxy[2]+xyxy[0])//2)
                cy = int((xyxy[3]+xyxy[1])//2)
                dis = (cx-640/2)**2+(cy-480/2)**2
                if dis < min_dis:
                    min_dis = dis
                    center_idx = len(obj_centers)
                    center_xy = [cx, cy]
                    print(dis)

                obj_centers.append([cx,cy]) # 중심

    result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45, 0.5)

    font = cv2.FONT_HERSHEY_PLAIN
    
    for i in range(len(bboxes)):
        label = str(CLASSES[int(class_ids[i][0])])
        bbox = list(map(int, bboxes[i])) 
        x, y, x2, y2 = bbox
        cx, cy = obj_centers[i]
        color = [255, 0, 0]

        if i == center_idx:
            color = [0, 255, 0]
            # depth = round((depth_info.get_distance(center_xy[0], center_xy[1]) * 100), 2)
            # wx, wy, wz = pyrealsense2.rs2_deproject_pixel_to_point(rs.depth_intrinsics, [cx, cy], depth)
            
            # wx = round(wx, 3)
            # wy = round(wy, 3)
            # wz = round(wz, 3)
            # cv2.putText(annotated_frame, "{}, {}, {}".format(wx, wy, wz), (x + 5, y + 60), 0, 1.0, color, 2)

        cv2.rectangle(annotated_frame, (x, y), (x2, y2), color, 2)

    resized_frame = cv2.resize(annotated_frame, (640, 480))
    cv2.imshow('Detecting pickle and tomato', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()