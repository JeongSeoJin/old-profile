import cv2
import numpy as np
import pickle
import pandas as pd
import time
import os
from datetime import datetime

# === 카메라 보정 데이터 불러오기 ===
file_path = 'C:\\Users\\sjl06\\Desktop\\Lab\\aruco marker\\camera_calibration.pkl'
with open(file_path, 'rb') as f:
    calibration_data = pickle.load(f)

camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# === ArUco 설정 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 15  # mm

# === 감마 조정 함수 ===
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# === 트랙바 설정 ===
cv2.namedWindow("Camera 1")
cv2.createTrackbar("Gamma x10", "Camera 1", 10, 30, lambda x: None)

# === 카메라 열기 ===
caps = [cv2.VideoCapture(2), cv2.VideoCapture(0)]
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === 실제 마커 거리 (카메라별) ===
true_lengths = {
    0: { (0,1):58,(1,2):58,(2,3):36,(3,4):58,(4,5):58,(5,6):36,(6,7):58,(7,8):58 },
    1: { (0,1):58,(1,2):66,(2,3):58,(3,4):58,(4,5):36,(5,6):58,(6,7):58,(7,8):58 }
}

# === 보정 계수 (거리용) ===
scales_per_camera = [{},{}]

# === 카메라별 캡처 데이터 ===
captured_data_per_camera = [[],[]]

# === 상대좌표 계산 함수 (각 마커 간 거리 성분 분해) ===
def calculate_relative_positions_mm(marker_positions, measured_lengths, origin_id):
    positions = {origin_id: np.array([0.0, 0.0])}
    sorted_ids = sorted(marker_positions.keys())

    for i in range(1, len(sorted_ids)):
        id_prev = sorted_ids[i-1]
        id_current = sorted_ids[i]
        pos_prev = positions[id_prev]

        # 마커 중심 좌표
        pt_prev = np.array(marker_positions[id_prev], dtype=float)
        pt_curr = np.array(marker_positions[id_current], dtype=float)

        # 방향 단위 벡터 (화면 좌표 Y 아래로 증가 → mm 좌표 반전)
        vec = pt_curr - pt_prev
        vec[1] = -vec[1]  # Y축 반전
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec_unit = np.array([1.0, 0.0])
        else:
            vec_unit = vec / norm

        # 실제 거리(mm)
        key = (min(id_prev,id_current), max(id_prev,id_current))
        d_mm = measured_lengths.get(key, norm)  # 보정 후 거리

        # X/Y 성분 분해하여 누적
        positions[id_current] = pos_prev + vec_unit * d_mm

    return positions

# === 카메라 프레임 처리 함수 ===
def process_camera(frame, gamma, camera_idx, update_scales=False):
    corrected = adjust_gamma(frame, gamma)
    corrected = cv2.undistort(corrected, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)
    marker_positions = {}
    measured_lengths = {}
    relative_positions_mm = {}

    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(corrected, corners, ids)

        # 마커 중심 계산
        for i, mid in enumerate(ids):
            corner_pts = corners[i].reshape(4,2)
            center_pt = np.mean(corner_pts, axis=0)
            px, py = int(round(center_pt[0])), int(round(center_pt[1]))
            marker_positions[mid] = (px, py)

        sorted_ids_list = sorted(ids)
        origin_id = 0 if 0 in sorted_ids_list else sorted_ids_list[0]

        # 인접 마커 거리 측정 및 보정
        for idx_pair in range(len(sorted_ids_list)-1):
            id1, id2 = sorted_ids_list[idx_pair], sorted_ids_list[idx_pair+1]
            t1, t2 = np.array(marker_positions[id1]), np.array(marker_positions[id2])
            measured = np.linalg.norm(t1 - t2)
            key = (min(id1,id2), max(id1,id2))
            true_length = true_lengths[camera_idx].get(key, None)
            if true_length and measured > 0:
                if update_scales or key not in scales_per_camera[camera_idx]:
                    scales_per_camera[camera_idx][key] = true_length / measured
                correction = scales_per_camera[camera_idx][key]
            else:
                correction = 1.0
            length_mm = measured * correction
            measured_lengths[key] = length_mm

            # 화면 표시 (거리 선)
            pt1, pt2 = marker_positions[id1], marker_positions[id2]
            mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
            cv2.line(corrected, pt1, pt2, (0,255,0), 2)
            cv2.putText(corrected, f'{length_mm:.1f}mm', mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # 상대 좌표 계산 (X/Y 성분 누적)
        relative_positions_mm = calculate_relative_positions_mm(marker_positions, measured_lengths, origin_id)

        # 화면 표시 (ID, 좌표, 중심점 색상 분리)
        for i, (mid, pos_mm) in enumerate(relative_positions_mm.items()):
            px, py = marker_positions[mid]

            # ID 텍스트
            cv2.putText(corrected, f'ID:{mid}', (px, py-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # 좌표 텍스트
            cv2.putText(corrected, f'({pos_mm[0]:.1f},{pos_mm[1]:.1f})mm', (px, py+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)

            # 마커 중심점
            cv2.circle(corrected, (px, py), 5, (0,0,255), -1)

    return corrected, relative_positions_mm, marker_positions

# === 메인 루프 ===
capture_mode = False
last_capture_time = 0
capture_interval = 0.5
capture_count = 0
session_time = None
output_dir = "captures"

while True:
    gamma_val = cv2.getTrackbarPos("Gamma x10","Camera 1")
    gamma = gamma_val/10.0 if gamma_val>0 else 0.1

    update_scales_flag = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord('i'):
        update_scales_flag = True
        print("[INFO] 보정계수 업데이트")

    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        processed, relative_positions_mm, marker_positions = process_camera(frame, gamma, idx, update_scales_flag)

        # 캡처 및 데이터 기록
        if capture_mode and (time.time() - last_capture_time >= capture_interval):
            capture_count += 1
            img_name = os.path.join(output_dir, f"{session_time}_cam{idx}_capture_{capture_count:03d}.png")
            cv2.imwrite(img_name, processed)

            for mid, pos_mm in relative_positions_mm.items():
                row = {"Frame": capture_count, "Marker ID": mid,
                       "X (mm)": round(float(pos_mm[0]),2),
                       "Y (mm)": round(float(pos_mm[1]),2)}
                captured_data_per_camera[idx].append(row)

            last_capture_time = time.time()
            print(f"[INFO] 카메라 {idx} - 캡처 {capture_count} 저장 완료")

        cv2.putText(processed, f'Gamma: {gamma:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        if capture_mode:
            cv2.putText(processed, "CAPTURING...", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        cv2.imshow(f"Camera {idx+1}", processed)

    # 키 입력 처리
    if key == ord('q'):
        break
    elif key == ord('a') and not capture_mode:
        capture_mode = True
        capture_count = 0
        captured_data_per_camera = [[],[]]
        last_capture_time = time.time()
        session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"captures_{session_time}"
        os.makedirs(output_dir, exist_ok=True)
        print("[INFO] 캡처 시작")
    elif key == ord('f') and capture_mode:
        capture_mode = False
        for cam_idx, data in enumerate(captured_data_per_camera):
            if data:
                df = pd.DataFrame(data)
                excel_name = os.path.join(output_dir, f"{session_time}_cam{cam_idx}_marker_data.xlsx")
                with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=f"Camera_{cam_idx}", index=False)
                print(f"[INFO] 카메라 {cam_idx} 엑셀 저장 완료: {excel_name}")

# 종료
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
