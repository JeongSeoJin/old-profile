import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        
        # -------------------------------------------------------------------
        # 1. 백본(Backbone) 설정: "이미지를 보는 눈" (은닉층 역할)
        # -------------------------------------------------------------------
        # 미리 학습된(pretrained=True) ResNet18을 불러옵니다.
        # 이 부분은 우리가 직접 학습시키는 게 아니라, 이미 똑똑해진 뇌를 빌려오는 '전이 학습' 부분입니다.
        # CNN의 복잡한 은닉층(Conv, Pooling 등)이 이 한 줄에 다 들어있습니다.
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # 백본의 마지막 출력 개수(ResNet18의 경우 1000개)를 자동으로 알아냅니다.
        # 즉, 그림 한 장을 보면 1000개의 숫자로 요약해서 뱉어낸다는 뜻입니다.
        out_features = list(self.backbone.modules())[-1].out_features

        # -------------------------------------------------------------------
        # 2. 분류기(Classifier Head): "비교하고 판단하는 머리"
        # -------------------------------------------------------------------
        # 두 이미지의 특징을 합친 후, 진짜 같은지 다른지 점수를 매기는 부분입니다.
        # 여러 층의 Linear(선형 결합)와 ReLU(활성화 함수)를 거치며 판단합니다.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),                  # 과적합 방지 (일부 뉴런 끄기)
            nn.Linear(out_features, 512),       # 1000개 특징 -> 512개로 압축
            nn.BatchNorm1d(512),                # 학습 안정화
            nn.ReLU(),                          # 활성화 함수 (비선형성 추가)

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),                 # 512개 -> 64개로 더 압축
            nn.BatchNorm1d(64),
            nn.Sigmoid(),                       # 값을 0~1 사이로 변환 (중간 단계)
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),                   # 최종적으로 숫자 1개 출력 (점수)
            nn.Sigmoid(),                       # 최종 확률값 (0.0 ~ 1.0)
                                                # 1.0에 가까우면 "같다", 0.0에 가까우면 "다르다"
        )

    def forward(self, img1, img2):
        # -------------------------------------------------------------------
        # 3. 순전파(Forward): 실제로 데이터가 흘러가는 과정
        # -------------------------------------------------------------------
        
        # [단계 1] 특징 추출 (Feature Extraction)
        # 두 이미지를 '동일한' 백본(ResNet)에 통과시킵니다.
        # *중요*: 두 개의 별도 네트워크가 아니라, 하나의 네트워크를 두 번 사용하는 것입니다. (가중치 공유)
        feat1 = self.backbone(img1)  # 이미지 A의 특징 벡터 (DNA)
        feat2 = self.backbone(img2)  # 이미지 B의 특징 벡터 (DNA)
        
        # [단계 2] 특징 결합 (Combination)
        # 두 특징 벡터를 요소별로 곱합니다(Element-wise Multiplication).
        # 두 이미지 모두에서 강하게 나타나는 특징은 큰 값이 되고, 하나라도 없으면 0이 됩니다.
        # 즉, "둘 다 가지고 있는 특징"을 강조하는 방식입니다.
        combined_features = feat1 * feat2

        # [단계 3] 최종 판단 (Classification)
        # 결합된 특징을 분류기(MLP)에 넣어 최종 점수(확률)를 계산합니다.
        output = self.cls_head(combined_features)
        
        return output