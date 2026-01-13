import cv2
import numpy as np
import matplotlib.pyplot as plt

import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from realsense.realsense_camera import DepthCamera

# 폴더 생성
folder_name = 'data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 리얼센스 카메라 초기화
rs = DepthCamera(1280, 720)

frame_count = 0

try:
    while True:
        ret, depth_raw_frame, color_raw_frame = rs.get_raw_frame()
        color_frame = np.asanyarray(color_raw_frame.get_data())
        depth_frame = np.asanyarray(depth_raw_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET)

        cv2.imshow('RealSense Camera', color_frame)

        key = cv2.waitKey(1)
        if key == ord(' '):  # 스페이스바를 누르면 이미지 저장
            frame_count += 1
            color = os.path.join(folder_name, f'color_{frame_count:04d}.png')
            depth = os.path.join(folder_name, f'depth_{frame_count:04d}.png')
            cv2.imwrite(color, color_frame)
            plt.imsave(depth, depth_frame)
            print(f'Saved {color, depth}')

        elif key == 27:  # ESC를 누르면 종료
            break

finally:
    rs.release()
    cv2.destroyAllWindows()