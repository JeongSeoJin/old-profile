import cv2
import glob

# 동영상 파일 경로
video_path = "/home/choiyj/Downloads/rviz_dataset/tomato_depth.mp4"

# 저장할 이미지 파일 경로
image_path = "/home/choiyj/Downloads/rviz_dataset/tomato/"

# 동영상 파일을 읽어옴
cap = cv2.VideoCapture(video_path)

# 프레임 수와 fps를 가져옴
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 0.1초 단위로 이미지 저장
interval = int(fps * 0.1)

# 이미지 저장할 폴더 생성
import os
if not os.path.exists(image_path):
    os.makedirs(image_path)

# 이미지 저장
for i in range(frame_count):
    ret, frame = cap.read()
    if i % interval == 0:
        cv2.imwrite(image_path + "depth_{:04d}.png".format(i), frame)
        print("save complete!")
        
# 동영상 파일 닫기
cap.release()