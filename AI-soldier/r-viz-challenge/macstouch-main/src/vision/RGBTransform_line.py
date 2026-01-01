import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 파일 불러오기
image_path = './data/imagecopy.png'  # 여기에 파일 경로를 입력하세요
image = cv2.imread(image_path)
image = cv2.resize(image, (480, 640))

yCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, Cr, Cb = cv2.split(yCrCb)
equalizedY = cv2.equalizeHist(y)
yCrCb2 = cv2.merge([equalizedY, Cr, Cb])
yCrCbDst = cv2.cvtColor(yCrCb2, cv2.COLOR_YCrCb2BGR)
blurred = cv2.GaussianBlur(yCrCbDst, (5, 5), 0)
yCrCbDst = cv2.Canny(blurred, 50, 100)

lines = cv2.HoughLinesP(yCrCbDst, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

roi1 = (160, 100, 150, 150)  # (x, y, width, height)
roi2 = (160, 365, 150, 150)

cv2.rectangle(image, (roi1[0], roi1[1]), (roi1[0]+roi1[2], roi1[1]+roi1[3]), (255, 0, 0), 2)
cv2.rectangle(image, (roi2[0], roi2[1]), (roi2[0]+roi2[2], roi2[1]+roi2[3]), (255, 0, 0), 2)

points_in_roi1 = 0
points_in_roi2 = 0

def get_line_pixels(x1, y1, x2, y2):
    line_pixels = []
    for x, y in zip(np.linspace(x1, x2, num=max(abs(x2-x1), abs(y2-y1))), np.linspace(y1, y2, num=max(abs(x2-x1), abs(y2-y1)))):
        line_pixels.append((int(x), int(y)))
    return line_pixels

for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y1 - y2) < 5:
        line_pixels = get_line_pixels(x1, y1, x2, y2)

        for (x, y) in line_pixels:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            # ROI 1에 점이 포함되는지 확인
            if roi1[0] <= x <= roi1[0] + roi1[2] and roi1[1] <= y <= roi1[1] + roi1[3]:
                points_in_roi1 += 1
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

            # ROI 2에 점이 포함되는지 확인
            if roi2[0] <= x <= roi2[0] + roi2[2] and roi2[1] <= y <= roi2[1] + roi2[3]:
                points_in_roi2 += 1
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

print('left' if points_in_roi1 > points_in_roi2 else 'right')
cv2.putText(image, "{}".format(points_in_roi1), (int(roi1[0] + roi1[2]/2), int(roi1[1] + roi1[3]/2)), 0, 1.0, (0, 255, 255), 2)
cv2.putText(image, "{}".format(points_in_roi2), (int(roi2[0] + roi2[2]/2), int(roi2[1] + roi2[3]/2)), 0, 1.0, (0, 255, 255), 2)
 

# 결과 시각화
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Horizontal Lines and ROI')
plt.show()