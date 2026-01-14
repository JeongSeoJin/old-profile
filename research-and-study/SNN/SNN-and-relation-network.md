# Relation Network Method

SLP (Distance-based): 고정된 기하학적 거리(Metric)에 의존해. 데이터가 선형적으로 잘 분리되는(Linearly Separable) 공간에 매핑되었다면 충분하지만, 과일처럼 조명, 각도, UOIS의 Segmentation 노이즈가 섞인 데이터는 그 공간이 찌그러져 있을(Manifold가 꼬여 있을) 확률이 높아.

MLP (Learnable Metric): 비선형 활성화 함수(ReLU 등)를 여러 번 거치면서, 단순한 '거리'가 아니라 두 이미지 사이의 **'관계(Relation)'**를 추론해. 즉, "거리가 가까우니 같다"가 아니라, "이런 특징들이 결합되니 같은 범주로 볼 수 있다"는 더 고차원적인 추론을 하는 거지.

차이 ($|h_1 - h_2|$): 두 벡터가 같을수록 0에 수렴해. 즉, '다름'을 측정하는 거야.곱하기 ($h_1 \otimes h_2$): Element-wise product는 일종의 Logical AND 연산과 비슷해. 두 이미지 모두에서 특정 feature가 강하게 발현될 때만 값이 커져.


Real-world Noise: 조명, 그림자, RealSense의 Depth 노이즈.

Segmentation Artifacts: UOIS가 완벽하지 않아 생기는 경계선 오차.

Augmentation: 일부러 데이터를 비틀어버림.

이런 상황에서 CNN이 뽑아낸 Feature Vector들은 Latent Space 상에서 선형적으로 분리되지 않고, 아주 복잡하게 꼬여 있었을 거야(Entangled Manifold). 이걸 풀어내려면 단순한 거리 측정이 아니라, **MLP라는 비선형 함수 근사기(Function Approximator)**가 필요했던 거지.

Element-wise Product ($h_1 \otimes h_2$): 이건 일종의 Hard Inductive Bias를 주는 거야.$$y_i = h_{1,i} \times h_{2,i}$$만약 $h_{1,i}$ (이미지 A의 i번째 특징, 예: 빨간색)가 크고, $h_{2,i}$ (이미지 B의 i번째 특징)도 크다면 $y_i$는 매우 커져. 둘 중 하나라도 작으면 0에 가까워지지.즉, **"두 이미지에서 공통적으로 활성화된 특징(Feature Matching)"**만을 강조해서 MLP에게 넘겨주는 역할을 해. 이게 Relation Network가 적은 데이터(Few-shot)로도 잘 동작하는 핵심 이유야. 너는 무의식적으로 이걸 선택했겠지만, 실제로는 아주 효율적인 Feature Selection 메커니즘을 사용한 셈이지.

---
## 요약 및 정리 
그럼 결국 **latent space**가 **embedding space** 랑 같은 개념이고 **latent vector**이 곧 **feature vector**이네. 그리고 latent space는 결국 데이터의 의미있는(중요한) 데이터만 남기면서 차원을 축소하는 역할이고. Manifold는 고차원적으로 보았을 때에는 네트워크가 직관적으로 분석할 수 없는 것, 즉, 유의미한 데이터로 사용할 수 없는 것이지만, 차원을 낮추어 분석하면 의미있는 데이터들을 얻을 수 있다는 것을 의미하는 거잖아. 이 고차원데이터들이 비선형적으로 **Manifold** 라는 굽은 평면 위에 plot되어있는 것이고, 우리는 그것을 펼쳐서 분석할 수 있는 거지.

- latent space : 차원 축소 & 의미 보존 
- Manifold: 고차원에 꼬여있는 데이터의 본질적 구조. 이걸 펴는 게 딥러닝의 핵심.
- 나의 구조: Element-wise Product로 특징 매칭 $\rightarrow$ MLP로 비선형 관계 추론.


**Metric Learning**은 분석하기 어려운 Manifold를 펼쳐서 Latent space로 대응시키는 거지. 그럼으로써 클래스를 나누기 어려웠던 것(분석이 어려운 것)도 클래스를 나눌 수 있게 되고.

결국 Siamese Network의 구조에서 CNN을 통해서 embedding space에 feature vector를 만들었는데 UOIS를 통해서 이미지를 segmentation하고 crop했지만 노이즈가 많기 때문에 단순히 이 feature vector에 L1 distance 방식으로 유사성을 판단하기에는 한계가 있는 거지. 

- CNN(encoder) : 입력받은 이미지가 Convolutional layer, pooling layer, FC layer, activation function을 통과하면서 차원이 축소되고, 특징이 뚜렸해짐(의미 보존) -> Latent Space로 재배열 하는 과정(이미지 그 자체의 모양인 Manifold를 펴서 좌표를 찍어줌)

- MLP(Relation Module) : 입력받은 두 feature vector의 결합을 통해 두 이미지 사이의 유사도를 판단(두 이미지의 닮음 여부인 Manifold펴서 좌표 찍기)


때문에 feature vector끼리 곱해서(element-wise product) logical AND연산 같은 것처럼, 유사한 특징은 극대화하고 아닌 특징은 낮춰주어 새로운 벡터를 만든 후에 이 데이터들이 존재하는 Manifold를 펼쳐서(MLP)를 거쳐서 비선형적인 관계를 학습하여 고차원의 데이터를 Latent space로 재배열시킴으로써 두 이미지 사이의 관계성(유사성)을 파악하여 유사도를 도출할 수 있는 거지.


### 자 내가 정리해볼게.

기존의 siamese network는 두 특징 벡터의 L1 distance를 구하고 단일 perceptron(SLP)인 FC layer를 통과해서 유사성을 판단함. 논문에서 다룬 데이터셋은 단순 희색 배경에 검정 문자이기 때문에 노이즈가 적고 gradient rate가 커서 linearly sperable함. 



하지만 realsense를 통해서 촬영한 사진을 UOIS(segmentation) + crop하여 학습하려다보니 조명, 색상, 등의 문제로 인해서 노이즈가 생겨서 단순히 논문에서 제시한 방식으로 해결하기엔 어려움.



때문에 더 고차원의 데이터를 추론할 수 있는 방식이 필요함. 



기존의 CNN을 통과한 후에 latent space의 벡터의 L1을 거리 차로 학습하는 것이 아닌 element wise product를 통해서 Logical And연산처럼 유사한 특징은 극대화하고 그렇지 않은 것의 특징은 약화해. 이후 MLP를 통해서 기존의 SLP가 해결하지 못하는 Non -linear decision boundary 를 찾아낼 수 있음. 즉, 기존의 방식과는 다르게 --relation network방식을 채택하여 분석하기 고차원적인 source data가 plot되어있는 manifold를 펴서 latent space에 다시 재배열함으로써 노이즈가 많은 데이터를 추론 및 분석하는 성능이 뛰어나다. 



추가로 Loss Function에 대한 중요한 특징이 존재한다. 논문의 경우 Contrasive Loss를 사용했다. 이는 Euclidean Space를 기반해야하는 Inductive bias라는 Contraint를 갖고 있다. 유사한 이미지의 경우에는 feature vector사이의 거리를 줄여야하고 유사하지 못한 이미지는 거리리가 멀도록 손실함수로 인해서 자연스레 학습을 한다. 하지만 이러한 강제성은 고차원의 데이터를 분석하기에 제한적이다.

반면에 BCE, MSE  Loss는 단순히 결과인 예측값과 정답만을 활용해서 MLP와 CNN의 가중치를 최적화하는 과정이다. 이떄 Contrasive Loss와는 다르게 Euclidean Space를 강제하지 않기 때문에 MLP가 feature vector를 더 효율적으로 분석하여 예측할 수 있도록 CNN가중치를 수정해 feature vector들이 비선형적인 패턴을 갖도록 학습이 된다. 이는 고차원적인 데이터를 학습하는데 이점이다. 



여기서 궁금한게 있어. feature vector들이 비선형적인 특성을 갖는다는 것이 살짝 헷갈려. feature vector이 비선형적이더라도 두 feature vector의 차이는 Euclidean Space에 있는 거 아니야? 그리고 Relation Module인 MLP가 어떻게 relation network의 원리를 활용해서 고차원적인 정보를 단일 MLP보다 더 잘 파악하는지 논리적, 수학적으로 궁금해.

