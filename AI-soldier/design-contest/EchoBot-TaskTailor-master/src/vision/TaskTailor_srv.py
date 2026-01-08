#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from std_msgs.msg import String
from soomac.srv import DefineTask, DefineTaskResponse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU 설정

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
from time import time
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# --- 핵심 커스텀 모듈 임포트 ---
# UOIS: 학습하지 않은 물체(Unseen Object)를 배경과 분리(Segmentation)하는 AI 모델
from uois.Uois import Uois
# utils: 분할된 마스크 정보를 바탕으로 이미지에서 물체 영역만 잘라내는(Crop) 함수
from utils.Seg2Crop import extract_objects_from_image
# Siamese: 두 이미지가 같은 물체인지 유사도를 비교하는 샴 네트워크 (One-shot Learning 추적용)
from siamese_network.eval import Siamese
from control.camera_transformation import transformation_define


folder_path = '/home/choiyoonji/catkin_ws/src/soomac/src/gui/Task/'

class GUI:
    def __init__(self) -> None:
        # ROS 서비스 서버 선언: 'define_task'라는 요청이 들어오면 path_callback 함수 실행
        name_sub = rospy.Service('define_task', DefineTask, self.path_callback)
        self.task_name = None

        # AI 모델 초기화 (메모리에 로드)
        print("Loading UOIS Model...")
        self.uois = Uois()
        print("Loading Siamese Network...")
        self.siamese = Siamese()

    def path_callback(self, req):
        """
        [핵심 로직 함수]
        사용자가 시연한 이미지 파일들을 순서대로 읽어서,
        물체들의 이동 경로를 추적하고 Pick & Place 작업을 정의합니다.
        """
        self.task_name = req.TaskName
        
        # 1. 저장된 시연 이미지(.npy 파일들) 불러오기
        # .npy 파일에는 보통 RGB 이미지와 Depth 정보가 함께 들어있거나 전처리된 데이터가 있습니다.
        npy_files = sorted(glob.glob(os.path.join(folder_path+self.task_name, '*.npy')))
        print(f"Loaded files: {npy_files}")

        object_list = [] # 첫 프레임에서 발견된 '기준 물체'들의 이미지 리스트
        coord_list = []  # 각 프레임(Step)별 물체들의 좌표 리스트 [[(x1,y1), (x2,y2)...], [...]]
        task_list = []   # 최종적으로 추출된 작업 단계(Pick/Place) 리스트
        
        # 각 단계(Step)별 작업 정보를 담을 딕셔너리 초기화
        step = {'pick': 0, 'place': 0} 
        task = {}

        # ------------------------------------------------------------------
        # [Step A] 모든 시연 프레임을 순회하며 물체 인식 및 추적 (Tracking)
        # ------------------------------------------------------------------
        for i, file in enumerate(npy_files):
            # 파일 로드 (RGB 이미지)
            img = np.load(file) 
            
            # --- 1. 객체 분할 (Segmentation) ---
            # UOIS 모델을 통해 이미지 내의 물체 마스크(seg) 추출
            rgb, seg = self.uois.run(img)
            
            # --- 2. 객체 추출 (Cropping) ---
            # 마스크 정보를 바탕으로 각 물체를 사각형 이미지로 잘라냄 (cropped_images)
            # coord: 해당 물체의 중심 좌표 (이미지 상 픽셀 좌표)
            cropped_images, coord = extract_objects_from_image(img, seg)

            # --- 3. 초기화 및 추적 (Tracking Logic) ---
            if i == 0:
                # [첫 번째 프레임인 경우] = 기준(Reference) 설정 단계
                # 이 프레임에 있는 물체들이 작업의 대상이 되는 '전체 물체 목록'이 됩니다.
                object_list = cropped_images
                
                # 디버깅 및 추후 확인을 위해 잘라낸 물체 이미지를 저장
                os.makedirs(folder_path+self.task_name+'/object', exist_ok=True)
                for idx, img in enumerate(object_list):
                    # object_list 구조: [ [중심좌표, 이미지배열], ... ] 로 추정됨
                    # img[1]이 실제 이미지 데이터
                    cv2.imwrite(folder_path+self.task_name+f'/object/object_{idx}.png', img[1])
            
            else:
                # [두 번째 프레임부터] = 추적(Matching) 단계
                # 이전 프레임의 물체(object_list)가 현재 프레임의 어디로 갔는지 찾습니다.
                coord = [] # 현재 프레임의 좌표들을 새로 담을 리스트
                
                # 기준 물체(object_list) 하나하나에 대해 현재 화면의 물체들과 비교
                for idx, img in enumerate(object_list):
                    # object_match 함수: 샴 네트워크를 이용해 가장 닮은 물체의 좌표를 반환
                    # img[1]: 기준 물체 이미지
                    # cropped_images: 현재 프레임에서 발견된 모든 물체 후보들
                    matched_coord = self.object_match(img[1], cropped_images)
                    coord.append(matched_coord)

            # 현재 프레임의 물체 위치들을 전체 리스트에 기록
            coord_list.append(coord)

        print('Object tracking complete. Analyzing movement...')
        
        task["coords"] = coord_list # 전체 좌표 궤적 저장
        coord_list = np.array(coord_list) # 계산을 위해 numpy 배열로 변환

        # ------------------------------------------------------------------
        # [Step B] 좌표 변화량을 분석하여 Pick & Place 추론 (Inference)
        # ------------------------------------------------------------------
        # 프레임(i)와 다음 프레임(i+1) 사이의 변화를 분석
        for i in range(len(npy_files)-1):
            
            # 1. Pick 감지 (가장 많이 움직인 물체 찾기)
            # 현재 프레임(i)과 다음 프레임(i+1) 사이의 모든 물체의 이동 거리 계산 (Euclidean Distance)
            dis = np.linalg.norm(coord_list[i] - coord_list[i+1], axis=1)
            
            # 이동 거리가 가장 큰(Max) 물체의 인덱스가 '집어 올려진(Pick)' 물체임
            pick_ind = np.argmax(dis)
            step['pick'] = int(pick_ind)

            # 2. Place 위치 감지 (놓여진 위치 찾기)
            # 여기서는 로직이 약간 특이함:
            # i번째 프레임의 물체들 좌표와, i+1번째 프레임에서 움직인(Pick된) 물체의 좌표 간 거리를 비교?
            # 통상적으로는 i+1번째 프레임에서 해당 물체의 좌표가 최종 Place 위치가 됨.
            # 이 코드는 Pick된 물체가 다음 프레임에서 어느 위치(기존 위치들 중 어디)와 가장 가까워졌는지 확인하려는 의도로 보임.
            # 혹은 타겟 위치(Place holder)가 이미 존재한다고 가정하고 그곳과의 매칭을 수행하는 것일 수 있음.
            dis = np.linalg.norm(coord_list[i] - coord_list[i+1][pick_ind], axis=1)
            place_ind = np.argmin(dis)
            step['place'] = int(place_ind)

            # 해당 스텝의 작업(Pick x번 -> Place y번 위치)을 리스트에 추가
            task_list.append(step.copy()) # .copy()를 써야 딕셔너리 참조 에러 방지

        task["steps"] = task_list
        print('Task definition complete.')

        # ------------------------------------------------------------------
        # [Step C] 결과 JSON 저장
        # ------------------------------------------------------------------
        # VisionNode가 읽을 수 있도록 .json 파일로 저장
        json_path = folder_path + self.task_name + '/' + self.task_name + '.json'
        with open(json_path, 'w') as json_file:
            json.dump(task, json_file, ensure_ascii=False, indent=4)

        print(f'JSON saved at: {json_path}')

        return DefineTaskResponse(True)

    def object_match(self, object_0, crop):
        """
        [Siamese Network 매칭 함수]
        One-shot Learning: 기준 이미지(object_0)와 
        현재 프레임의 후보 이미지들(crop)을 비교하여 
        가장 유사도가 높은 물체의 좌표를 반환합니다.
        """
        max_score = -1
        best_coord = [0,0] # 못 찾았을 경우 기본값
        
        # 현재 프레임에서 잘라낸 모든 물체 후보들을 순회
        for img in crop:
            # img 구조: [중심좌표, 이미지배열]
            
            # 샴 네트워크를 통해 유사도 점수(0~1) 계산
            # self.siamese.eval(기준이미지, 비교대상)
            similarity = self.siamese.eval(object_0, img[1])
            
            # 가장 높은 유사도를 가진 물체를 선택 (Argmax Logic)
            if similarity > max_score:
                max_score = similarity
                best_coord = img[0] # 그 물체의 좌표 저장

        return best_coord

# --- 메인 실행부 ---
if __name__ == "__main__":
    rospy.init_node('task_tailor_service') # 노드 이름 설정
    server = GUI() # 서비스 클래스 인스턴스 생성
    print("TaskTailor Service is ready.")
    rospy.spin() # 서비스 요청 대기 (무한 루프)


