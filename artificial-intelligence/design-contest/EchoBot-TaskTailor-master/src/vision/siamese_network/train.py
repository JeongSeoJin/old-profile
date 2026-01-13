import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 직접 만든 model.py와 dataset.py를 가져옵니다.
from model import SiameseNetwork
from dataset import Dataset

if __name__ == "__main__":
    # ==========================================
    # 1. 환경 설정 및 하이퍼파라미터 (훈련 계획 세우기)
    # ==========================================
    
    # 데이터가 저장된 경로 설정 (훈련용 / 검증용)
    train_path = "/home/choiyoonji/catkin_ws/src/soomac/src/vision/siamese_network/data/train"
    val_path = "/home/choiyoonji/catkin_ws/src/soomac/src/vision/siamese_network/data/val"
    
    # 결과물(학습된 모델 파일)을 저장할 경로
    out_path = "/home/choiyoonji/catkin_ws/src/soomac/src/vision/siamese_network"
    
    # 사용할 백본 네트워크 (ResNet18: 빠르고 성능 좋은 눈)
    backbone = "resnet18"
    
    # 학습률 (Learning Rate): 한 번에 얼마나 많이 배울지 (너무 크면 대충 배우고, 너무 작으면 느림)
    learning_rate = 1e-4  
    
    # 에폭 (Epochs): 문제집 한 권을 총 몇 번 반복해서 풀 것인가? (1000번 반복)
    epochs = 1000
    
    # 모델 저장 주기: 1000번 중 10%인 100번에 한 번씩은 무조건 저장하겠다.
    save_after = int(epochs/10)

    # 저장할 폴더가 없으면 새로 만듭니다.
    os.makedirs(out_path, exist_ok=True)

    # GPU가 있으면 쓰고, 없으면 CPU를 씁니다. (GPU가 수십 배 빠릅니다)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    # 2. 데이터 준비 (공부할 문제집 펴기)
    # ==========================================
    
    # Dataset: 폴더에서 이미지를 읽어와서 정답/오답 쌍(Pair)을 만드는 역할
    # augment=True: 훈련할 때는 이미지를 비틀고 돌려서 어렵게 공부시킴 (응용력 키우기)
    train_dataset   = Dataset(train_path, shuffle_pairs=True, augment=True)
    
    # augment=False: 시험(검증) 볼 때는 원본 그대로 평가함
    val_dataset     = Dataset(val_path, shuffle_pairs=False, augment=False)
    
    # DataLoader: 데이터를 8개씩(batch_size) 묶어서 모델에게 배달해주는 역할
    train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=8)

    # ==========================================
    # 3. 모델 및 학습 도구 준비 (학생, 코치, 채점표)
    # ==========================================
    
    # 학생: 샴 네트워크 모델을 생성하고 GPU로 이동시킵니다.
    model = SiameseNetwork(backbone=backbone)
    model.to(device)

    # 코치(Optimizer): 틀린 만큼 가중치(뇌세포)를 수정해주는 역할 (Adam 사용)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 채점표(Loss Function): 정답(0 또는 1)과 예측값(확률)의 차이를 계산 (이진 분류 손실)
    criterion = torch.nn.BCELoss()

    # 기록장(Tensorboard): 학습 과정을 그래프로 그려주는 도구 설정
    writer = SummaryWriter(os.path.join(out_path, "summary"))

    # 최고 점수 기록용 변수 (초기값은 아주 큰 숫자로 설정)
    best_val = 10000000000

    # ==========================================
    # 4. 학습 루프 시작 (본격적인 훈련)
    # ==========================================
    for epoch in range(epochs):
        print("[{} / {}]".format(epoch, epochs))
        
        # [중요] 모델을 '훈련 모드'로 전환 (Dropout 켜기, Batch Norm 학습 모드)
        model.train()

        losses = []  # 이번 에폭의 오차들을 담을 리스트
        correct = 0  # 맞춘 개수
        total = 0    # 전체 문제 개수
   
        # DataLoader에서 데이터 뭉치(Batch)를 하나씩 꺼내옵니다.
        for (img1, img2), y, (class1, class2) in train_dataloader:
            # 데이터를 GPU로 옮깁니다. (map 함수로 한 번에 처리)
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            # 1. 예측 (Forward): 두 이미지를 모델에 넣고 확률(prob)을 얻습니다.
            prob = model(img1, img2)
            
            # 2. 채점 (Loss Calculation): 예측 확률과 정답(y)의 차이(오차)를 계산합니다.
            loss = criterion(prob, y)

            # 3. 초기화 (Zero Grad): [매우 중요] 이전 배치의 기울기 계산 값을 지웁니다.
            optimizer.zero_grad()
            
            # 4. 역전파 (Backward): 오차를 역추적해서 각 뉴런이 얼마나 잘못했는지 찾아냅니다.
            loss.backward()
            
            # 5. 수정 (Step): 찾은 잘못만큼 가중치를 아주 조금 수정합니다.
            optimizer.step()

            # --- 기록용 코드 ---
            losses.append(loss.item()) # 오차 기록
            # 맞춘 개수 계산: 확률이 0.5 넘으면 1(같음), 아니면 0(다름)으로 판단
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)
    
        # 한 에폭이 끝나면 학습 결과(오차 평균, 정확도)를 텐서보드에 적습니다.
        writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('train_acc', correct / total, epoch)

        # 화면에 현재 성적 출력
        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

        # ==========================================
        # 5. 검증 루프 (모의고사) - 학습하지 않고 평가만 함
        # ==========================================
        
        # [중요] 모델을 '평가 모드'로 전환 (Dropout 끄기, Batch Norm 고정)
        model.eval()

        losses = []
        correct = 0
        total = 0

        # 검증용 데이터로 채점 (optimizer.step() 없음 -> 학습 안 함)
        for (img1, img2), y, (class1, class2) in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)    # 문제 풀기
            loss = criterion(prob, y)   # 채점 하기

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        # 검증 결과 계산
        val_loss = sum(losses)/max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))

        # ==========================================
        # 6. 모델 저장 (성적표 관리)
        # ==========================================

        # "어? 이번 검증 점수(Loss)가 역대 최저(best_val)보다 낮네?" (신기록 달성)
        if val_loss < best_val:
            best_val = val_loss # 최고 기록 갱신
            # 'best.pth'라는 이름으로 모델 파일 저장 (eval.py에서 이걸 씁니다!)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(), # 핵심: 가중치 값들
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "best.pth")
            )            

        # 정기 저장: 점수가 좋든 나쁘든 정해진 주기(save_after)마다 백업 저장
        if (epoch + 1) % save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "epoch_{}.pth".format(epoch + 1))
            )