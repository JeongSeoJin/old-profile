import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        # 1. ResNet18 모델을 통째로 가져옵니다. 
        # pretrained=True: 이미 ImageNet(100만 장 사진)으로 공부를 마친 똑똑한 뇌를 가져옵니다. > 이미 배운 것을 활용(전이학습)
        # 추후에 train.py에서는 이 지식들을 바탕으로 어떻게 feature를 뽑아내어 비교/판별하는지 학습한다.
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # 2. 백본의 마지막 레이어가 몇 개의 숫자를 뱉어내는지 확인합니다.
        # ResNet18의 경우, 기본적으로 1000개(ImageNet 클래스 개수)를 출력합니다.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same object or different.
        self.cls_head = nn.Sequential(
            # [Layer 1] 입력층 -> 은닉층 (512개 뉴런)
            nn.Dropout(p=0.5),              # 과적합 방지 (일부 뉴런을 끔)
            nn.Linear(out_features, 512),   # 1000개 특징 -> 512개로 압축
            nn.BatchNorm1d(512),            # 학습 안정화 (데이터 분포 정렬) > 0 근처의 값으로 만들어줌
            nn.ReLU(),                      # 활성화 함수 (음수는 0으로)

            # [Layer 2] 은닉층 -> 은닉층 (64개 뉴런)
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),             # 512개 -> 64개로 더 압축 (엑기스만 남김)
            nn.BatchNorm1d(64),
            nn.Sigmoid(),                   # 특이점: 중간에 Sigmoid를 씀 (값을 0~1로 누름)

            # [Layer 3] 은닉층 -> 출력층 (1개 뉴런)
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),               # 64개 -> 최종 점수 1개
            nn.Sigmoid(),                   # 최종 확률 출력 (0.0 ~ 1.0)
)

    def forward(self, img1, img2):
        # Step 1: 특징 추출 (Feature Extraction)
        feat1 = self.backbone(img1)  # 이미지 A -> 1000개의 숫자
        feat2 = self.backbone(img2)  # 이미지 B -> 1000개의 숫자
        
        # Step 2: 특징 결합 (Feature Fusion) - ★ 핵심 ★
        combined_features = feat1 * feat2

        # Step 3: 최종 판결 (Classification)
        output = self.cls_head(combined_features)
        return output