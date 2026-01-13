# ---------------------------------------
# 1. 가상 데이터셋 생성기 (Run this first!)
# ---------------------------------------
import os
import cv2
import numpy as np
from tqdm import tqdm

# [1] 폴더 구조 잡기
# 데이터를 저장할 'data' 폴더와 그 안에 'train(공부용)', 'val(모의고사용)' 폴더를 만듭니다.
base_path = "./data"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")

classes = ["circle", "rectangle"] # 우리가 구분할 두 가지 물체

# 폴더가 없으면 새로 만듭니다. (exist_ok=True: 이미 있으면 에러 안 내고 넘어감)
for path in [train_path, val_path]:
    for cls in classes:
        os.makedirs(os.path.join(path, cls), exist_ok=True)

print("데이터 폴더 생성 완료! 이미지를 그립니다...")

# [2] 이미지를 그리는 화가 함수
def create_dummy_data(root_path, count=100):
    for i in range(count):
        # 100x100 크기의 검은색(0) 도화지를 준비합니다. (채널 3: 컬러 모드)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 'circle' 폴더에 저장할 경우 -> 동그라미 그리기
        if "circle" in root_path:
            # 위치(center)와 크기(radius)를 랜덤으로 정해서 다양성을 줍니다.
            center = (np.random.randint(30, 70), np.random.randint(30, 70))
            radius = np.random.randint(10, 30)
            color = (255, 255, 255) # 흰색
            # cv2.circle(이미지, 중심, 반지름, 색상, 두께(-1은 채우기))
            cv2.circle(img, center, radius, color, -1)
            save_path = os.path.join(root_path, "circle", f"circle_{i}.jpg")
            
        # 'rectangle' 폴더에 저장할 경우 -> 네모 그리기
        else:
            # 좌측 상단(pt1)과 우측 하단(pt2) 좌표를 랜덤으로 찍습니다.
            pt1 = (np.random.randint(10, 40), np.random.randint(10, 40))
            pt2 = (np.random.randint(60, 90), np.random.randint(60, 90))
            color = (255, 255, 255)
            cv2.rectangle(img, pt1, pt2, color, -1)
            save_path = os.path.join(root_path, "rectangle", f"rect_{i}.jpg")
            
        # 완성된 그림을 파일로 저장합니다.
        cv2.imwrite(save_path, img)

# 훈련용 100장, 검증용 20장씩 생성 (너무 적으면 학습이 안 되니 테스트용으로 적당함)
# *팁: 실전에서는 이 숫자를 늘려주세요!
create_dummy_data(train_path, 100)
create_dummy_data(val_path, 20)
print("✅ 가상 데이터셋 생성 완료!")

#################################################################################

# ---------------------------------------
# 2. 학습 실행 (Training)
# ---------------------------------------
import torch
from torch.utils.data import DataLoader
from model import SiameseNetwork
from dataset import Dataset
import matplotlib.pyplot as plt

# --- [설정값] 하이퍼파라미터 ---
train_dir = "./data/train"
val_dir = "./data/val"
BATCH_SIZE = 16  # 한 번에 16문제씩 풀겠다
EPOCHS = 10      # 문제집을 처음부터 끝까지 10번 반복해서 보겠다

# GPU가 있으면 GPU를, 없으면 CPU를 사용 (속도 차이가 큽니다!)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 장치: {device}")

# [1] 데이터 로더 준비 (문제 출제 위원)
# shuffle_pairs=True: 훈련 때는 매번 랜덤하게 짝을 지어줘서 꼼수를 못 쓰게 함
train_dataset = Dataset(train_dir, shuffle_pairs=True, augment=True)
val_dataset = Dataset(val_dir, shuffle_pairs=False, augment=False)

# DataLoader: 데이터를 배치 단위(16개)로 묶어서 배달해주는 역할
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# [2] 모델 준비 (뇌)
# ResNet18 백본을 가진 샴 네트워크 생성 후, GPU로 이사시킴
model = SiameseNetwork(backbone="resnet18")
model.to(device)

# Optimizer: 틀린 문제를 보고 뇌세포(가중치)를 어떻게 고칠지 결정하는 도구 (Adam 추천)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Criterion: 채점 기준표 (BCE Loss = 이진 분류용 오차 측정 함수)
# 1.0이어야 하는데 0.8이라고 하면 오차(Loss)가 얼마인지 계산
criterion = torch.nn.BCELoss()

# 학습 과정을 그래프로 그리기 위해 점수 기록장
loss_history = []

print("🚀 학습 시작!")
for epoch in range(EPOCHS):
    model.train() # "자, 공부 시작! (학습 모드)" - Dropout 등이 켜짐
    epoch_loss = []
    
    # 데이터 로더에서 문제 꾸러미(이미지 2장, 정답)를 하나씩 꺼냄
    for (img1, img2), y, _ in train_loader:
        # 데이터를 GPU로 보냄
        img1, img2, y = img1.to(device), img2.to(device), y.to(device)
        
        # [핵심 학습 루프]
        optimizer.zero_grad()       # 1. 이전에 계산했던 기울기 초기화 (깨끗한 상태)
        output = model(img1, img2)  # 2. 시험 봄 (순전파) -> 예측값 나옴
        loss = criterion(output, y) # 3. 채점 함 (오차 계산)
        loss.backward()             # 4. 오답 노트 작성 (역전파) -> 어디를 고쳐야 할지 계산
        optimizer.step()            # 5. 뇌 수정 (가중치 업데이트)
        
        epoch_loss.append(loss.item())
    
    # 한 에폭(Epoch)의 평균 점수 계산
    avg_loss = sum(epoch_loss) / len(epoch_loss)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

print("✅ 학습 완료!")

# [3] 결과 확인 및 저장
# Loss가 뚝뚝 떨어져야 정상입니다. (예: 0.6 -> 0.1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 다 배운 모델을 파일로 저장 (나중에 다시 쓰려고)
torch.save(model.state_dict(), "siamese_colab.pth")

#################################################################################
import os
import random
from PIL import Image
from torchvision import transforms

# --- 테스트 설정 ---
model_path = "siamese_colab.pth"

# [1] 이미지 전처리 규칙
# *주의*: 학습할 때랑 똑같은 규칙(크기, 정규화 값)으로 해야 합니다.
# 학습 때는 100x100으로 배웠는데 테스트 때 200x200을 주면 헷갈려합니다.
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# [2] 저장된 모델 불러오기
model = SiameseNetwork(backbone="resnet18")
# map_location: GPU에서 저장한 걸 CPU에서 불러올 때도 에러 안 나게 처리
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# ★ 중요: 평가 모드로 전환!
# 이걸 안 하면 Dropout 같은 기능이 켜져 있어서 매번 결과가 달라질 수 있음
model.eval() 

# 테스트용 랜덤 이미지 하나 뽑는 함수
def get_random_image(class_name):
    path = f"./data/val/{class_name}"
    files = os.listdir(path)
    img_name = random.choice(files)
    img_path = os.path.join(path, img_name)
    return Image.open(img_path).convert("RGB")

# [3] 비교 및 시각화 함수
def compare_images(img1, img2, title_text):
    # 전처리 후 차원 확장
    # 모델은 (Batch, Channel, H, W) 형태를 원하는데, 이미지는 (C, H, W)임.
    # unsqueeze(0)으로 맨 앞에 1차원을 추가해서 (1, 3, 100, 100)으로 만듦
    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)

    # 평가 때는 기울기 계산 불필요 (메모리 절약)
    with torch.no_grad():
        score = model(t1, t2).item() # 결과값(Tensor)을 숫자(Float)로 변환

    # 결과 그림 그리기
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img1)
    axes[0].set_title("Image A")
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title("Image B")
    axes[1].axis('off')

    # 점수 출력 (1.0에 가까우면 같다, 0.0에 가까우면 다르다)
    plt.suptitle(f"{title_text}\nSimilarity: {score:.4f}", fontsize=14, color='blue', y=1.05)
    plt.tight_layout()
    plt.show()

# --- 실제 테스트 실행 ---
print("🧪 테스트 1: 같은 모양 비교 (동그라미 vs 동그라미)")
imgA = get_random_image("circle")
imgB = get_random_image("circle")
compare_images(imgA, imgB, "[Same Pair]") # 예상: 0.9 이상 나와야 함

print("\n🧪 테스트 2: 다른 모양 비교 (동그라미 vs 네모)")
imgC = get_random_image("circle")
imgD = get_random_image("rectangle")
compare_images(imgC, imgD, "[Diff Pair]") # 예상: 0.1 이하 나와야 함