# RNN계열의 LSTM 모델을 사용하여 제스처 인식 모델(연속된 동작을 인식하기 위한)을 학습시키기 위한 데이터셋 생성 코드입니다.
# CNN 모델과 달리 RNN 모델은 시퀀스 데이터를 필요로 하므로, 각 제스처에 대해 일정 길이의 시퀀스 데이터를 생성합니다.
# 이 코드는 MediaPipe를 사용하여 손 관절 데이터를 추출하고, 각 제스처에 대해 시퀀스 데이터(LSTM의 특성)를 생성하여 저장합니다.
# LSTM는 시퀀스 데이터를 필요로 하므로, 각 제스처에 대해 일정 길이의 시퀀스 데이터를 생성합니다. 시퀀스 데이터는 연속인 데이터 프레임을 구성합니다.
# 즉, 제스처의 속도와 방향을 분석하여 학습에 도움을 줍니다.
#############DEFENSE CODE#############
# 다른 무거운 3D-CNN 모델 대신 가벼운 RNN 모델계열인 LSTM을 사용하여 실시간 제스처 인식이 가능하도록 설계되었습니다.
# 군대에서는 소리를 낼 수 없는 은밀한 작전을 수행할 때, 손 제스처를 통해 명령을 전달하는 것이 중요합니다. LSTM 기반 제스처인식을 통한 로봇 제어는
# 이러한 상황에서 '유무인 복합 전투체계'에서 효율적으로 인간과 로봇의 소통을 더 효율적으로 가능하게 합니다.


import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['come', 'away', 'spin']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4)) # 21개 손의 특징점 : x,y,z,visibility
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint 손목에서 이어지는 시작점들
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint 손가락 끝점들
                    v = v2 - v1 # [20, 3] 각 관절 간의 벡터 계산
                    # Normalize v 정규화를 통해서 손의 크기의 영향을 없앰
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) #idx : actions의 인덱스 라벨

                    d = np.concatenate([joint.flatten(), angle_label]) # 21개 관절의 x,y,z,visibility + 15개 각도 + 1개 라벨 = 100차원
                                                                        #concatenate : 배열을 하나로 합침

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
