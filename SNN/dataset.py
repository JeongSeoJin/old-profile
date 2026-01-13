import os
import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# IterableDataset을 상속받습니다. (데이터를 미리 다 메모리에 올리지 않고, 필요할 때 하나씩 꺼내 쓰는 방식)
class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=True, augment=False):
        '''
        데이터셋 초기화 함수
        - path: 데이터가 있는 폴더 경로 (예: ./data/train)
        - shuffle_pairs: 훈련할 때는 True(랜덤), 검증할 때는 False(고정)
        - augment: 데이터 증강 여부 (훈련할 때 True로 해서 데이터를 꼬아줍니다)
        '''
        self.path = path
        self.feed_shape = [3, 100, 100]  # 입력 이미지 크기 (채널 3, 높이 100, 너비 100)
        self.shuffle_pairs = shuffle_pairs
        self.augment = augment

        # -------------------------------------------------------------------
        # [1] 이미지 변환 설정 (Transforms)
        # -------------------------------------------------------------------
        if self.augment:
            # 훈련용: 이미지를 일부러 찌그러트리고 뒤집어서 어렵게 만듭니다. (강한 모델을 만들기 위해)
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2), # 회전, 이동, 확대/축소, 비틀기
                transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전
                transforms.ToTensor(), # 이미지를 0~1 사이의 숫자 텐서로 변환
                # 정규화: ResNet이 학습했던 데이터와 비슷한 분포(평균, 표준편차)로 맞춰줌
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:]) # 크기 강제 조절 (100x100)
            ])
        else:
            # 테스트/검증용: 이미지를 깨끗한 상태 그대로 씁니다. (평가는 정직하게!)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])

        # 초기 실행 시 한 번 짝을 만들어 둡니다.
        self.create_pairs()

    # -------------------------------------------------------------------
    # [2] 짝 만들기 함수 (가장 중요!)
    # -------------------------------------------------------------------
    def create_pairs(self):
        # 폴더 내의 모든 jpg 파일 경로를 가져옵니다.
        self.image_paths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        self.image_classes = []
        self.class_indices = {} # 예: {'circle': [0, 1, 5...], 'rectangle': [2, 3, 4...]}

        # 파일들을 훑으면서 "어떤 이미지가 어떤 클래스(circle/rect)인지" 족보를 만듭니다.
        for image_path in self.image_paths:
            # 경로 이름에서 클래스 이름 추출 (data/train/circle/img1.jpg -> circle)
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            # 해당 클래스 리스트에 이미지의 번호(index)를 저장
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        # 첫 번째 이미지들의 번호 리스트 (0번부터 끝까지)
        self.indices1 = np.arange(len(self.image_paths))

        # 훈련 시에는 매번 랜덤하게 섞습니다.
        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # 검증 시에는 결과를 똑같이 유지하기 위해 시드 고정
            np.random.seed(1)

        # -------------------------------------------------------------------
        # ★ 핵심 로직: 50% 확률로 같은 그림(1) or 다른 그림(0) 결정
        # -------------------------------------------------------------------
        # select_pos_pair가 True면 '같은 종류', False면 '다른 종류'를 짝지어 줄 예정
        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        # 첫 번째 이미지(i)와 운명(pos: 같음/다름)을 하나씩 꺼내서 짝꿍을 정해줍니다.
        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i] # 첫 번째 이미지의 종류
            
            if pos:
                # [Positive Pair] 운명이 '같음'이라면 -> 같은 종류(class1)를 선택
                class2 = class1
            else:
                # [Negative Pair] 운명이 '다름'이라면 -> 현재 종류를 뺀 나머지 중에서 랜덤 선택
                # set 연산: 전체 클래스 집합 - 현재 클래스 = 다른 클래스들
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
            
            # 결정된 class2 목록 중에서 이미지 하나를 무작위로 뽑음 (idx2)
            idx2 = np.random.choice(self.class_indices[class2])
            self.indices2.append(idx2)
            
        self.indices2 = np.array(self.indices2)

    # -------------------------------------------------------------------
    # [3] 데이터 배달 (Iterator)
    # -------------------------------------------------------------------
    def __iter__(self):
        # 에폭(Epoch)마다 새로 짝을 짓습니다. (매번 다른 문제 풀게 하기 위함)
        self.create_pairs()

        # 만들어둔 짝꿍 리스트(idx, idx2)를 순서대로 꺼냅니다.
        for idx, idx2 in zip(self.indices1, self.indices2):

            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            # 이미지 파일을 열어서 RGB로 변환
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            # 전처리(크기조절, 텐서변환 등) 수행
            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            # 최종 배달: (이미지1, 이미지2), 정답(1=같음, 0=다름), (클래스이름1, 클래스이름2)
            yield (image1, image2), torch.FloatTensor([class1==class2]), (class1, class2)
        
    def __len__(self):
        return len(self.image_paths)