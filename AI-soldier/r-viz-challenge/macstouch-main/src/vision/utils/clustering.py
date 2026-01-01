import cv2
import numpy as np
import open3d as o3d

import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from realsense.utils import compute_xyz

# 카메라 내부 파라미터 (예: focal length와 principal point)
camera_intrinsics = {'fx': 387.5052185058594, 'fy': 387.5052185058594, 'x_offset': 324.73431396484375, 'y_offset': 238.08770751953125, 'img_height': 480, 'img_width': 640}
fx = camera_intrinsics['fx']
fy = camera_intrinsics['fy']
cx = camera_intrinsics['x_offset']
cy = camera_intrinsics['y_offset']

depth_scale = 1000

# 깊이 이미지 불러오기 (16비트 PNG 또는 다른 형식)
depth_image = cv2.imread('/home/choiyj/Downloads/rviz_dataset/tomato/depth_0210.png', cv2.IMREAD_GRAYSCALE)
depth = compute_xyz(depth_image * depth_scale,camera_intrinsics).reshape((-1,3))

# Open3D를 사용해 Point Cloud 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(depth)

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd])
