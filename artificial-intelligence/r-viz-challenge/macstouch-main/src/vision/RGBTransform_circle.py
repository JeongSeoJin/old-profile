import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 파일 불러오기
image_path = './data/IMG_2803.png'  # 여기에 파일 경로를 입력하세요
image = cv2.imread(image_path)

# BGR 채널을 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# R, G, B 채널 분리
R, G, B = cv2.split(image_rgb)

# 각 채널을 그레이스케일로 변환
R_gray = R
G_gray = G
B_gray = B

# 오리지널 이미지 그레이스케일 변환 (R, G, B 평균값을 사용)
original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection 적용
R_edges = cv2.Canny(R_gray, 100, 200)
G_edges = cv2.Canny(G_gray, 100, 200)
B_edges = cv2.Canny(B_gray, 100, 200)
original_edges = cv2.Canny(original_gray, 100, 200)

# 허프 변환을 이용한 원 검출 함수
def detect_circles(image_edges, original_image):
    circles = cv2.HoughCircles(image_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=180, maxRadius=250) # 파라미터 튜닝 필요

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(original_image, (x, y), r, (0, 255, 0), 4)  # 원 그리기
            cv2.rectangle(original_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # 중심점 표시
    return original_image

# R, G, B 채널과 오리지널 이미지에서 원 검출
R_circles = detect_circles(R_edges, cv2.merge([R, np.full_like(R, 255), np.full_like(R, 255)]))
G_circles = detect_circles(G_edges, cv2.merge([np.full_like(G, 255), G, np.full_like(G, 255)]))
B_circles = detect_circles(B_edges, cv2.merge([np.full_like(B, 255), np.full_like(B, 255), B]))
original_circles = detect_circles(original_edges, image_rgb.copy())  # 원래 이미지에도 허프 변환 적용

# 결과 출력
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(R_circles)
plt.title('Red Channel (Hough Circles)')

plt.subplot(2, 2, 2)
plt.imshow(G_circles)
plt.title('Green Channel (Hough Circles)')

plt.subplot(2, 2, 3)
plt.imshow(B_circles)
plt.title('Blue Channel (Hough Circles)')

plt.subplot(2, 2, 4)
plt.imshow(original_circles)
plt.title('Original Image (Hough Circles)')

plt.tight_layout()
plt.show()
