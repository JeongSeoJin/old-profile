import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# RGB와 Depth 스트림을 활성화
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프라인 시작
pipeline.start(config)

# 비디오 파일 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_out = cv2.VideoWriter('tomato_rgb.mp4', fourcc, 30, (640, 480))
depth_out = cv2.VideoWriter('tomato_depth.mp4', fourcc, 30, (640, 480))

try:
    while True:
        # 프레임 데이터 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Depth 이미지 정규화 및 컬러 맵 적용 (시각화 용도)
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_colored = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # 비디오 파일에 프레임 저장
        rgb_out.write(color_image)
        depth_out.write(depth_image_colored)

        # 이미지 표시 (옵션)
        cv2.imshow('RGB Image', color_image)
        cv2.imshow('Depth Image', depth_image_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 비디오 파일과 파이프라인 정지
    rgb_out.release()
    depth_out.release()
    pipeline.stop()
    cv2.destroyAllWindows()
