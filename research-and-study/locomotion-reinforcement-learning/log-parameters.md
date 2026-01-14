## Reinforcement Learning 학습 시 parameters 의미
| rollout/ | |

| ep_len_mean | 980 |

| ep_rew_mean | 3.96e+03 |

| time/ | |

| fps | 326 |

| iterations | 472 |

| time_elapsed | 2959 |

| total_timesteps | 966656 |

| train/ | |

| approx_kl | 0.09200071 |

| clip_fraction | 0.536 |

| clip_range | 0.2 |

| entropy_loss | -32.2 |

| explained_variance | 0.516 |

| learning_rate | 0.0003 |

| loss | -0.451 |

| n_updates | 4710 |

| policy_gradient_loss | -0.077 |

| std | 3.59 |

| value_loss | 0.0034 |

rollout/ep_len_mean : 로봇이 한 번 시작해서 몇 스텝이나 버텼는지의 평균.

rollout/ep_rew_mean : 한 판(Episode) 동안 받은 점수의 총합 평균.

train/entropy_loss : 행동의 무작위성(Randomness). 절댓값이 클수록 더 다양한(혹은 더 과격한) 시도를 한다는 뜻.

train/explained_variance : AI(Critic)가 점수를 얼마나 잘 예측하는지. (1.0 = 신, 0 = 찍기, 음수 = 바보)

train/std : 로봇이 행동을 결정할 때 더하는 "노이즈(떨림)의 크기".

train/value_loss: 점수 예측이 얼마나 틀렸는지(오차). 이 값은 낮을수록 좋다. 학습이 잘 될수록 줄어든다.

approx_kl : 이번 업데이트로 뇌가 얼마나 많이 변했는지. 0.02 - 0.005 사이 값을 유지

clip_fraction : 얼만큼 restrict를 걸고 있는지 0.2 이하의 값이 좋음

clip_range : PPO알고리즘의 clipping 범위

성공적인 학습 시나라오
ep_rew_mean: 처음엔 살짝 떨어지거나 천천히 오르지만, 꾸준히 상승함.

approx_kl: 0.005 ~ 0.02 사이 유지.

clip_fraction: 0.2 미만 유지.

std: 1.0 ~ 0.8 수준으로 서서히 감소.

판단: "그대로 쭉 밀고 가세요(Keep going)." 100만 스텝 이후엔 아주 훌륭한 워킹을 볼 수 있습니다.

## std
1. 🧠 뇌(Policy)의 출력 구조: "평균"과 "분산"PPO 같은 알고리즘에서 Policy 네트워크는 액션 하나(예: "0.5만큼 움직여")를 딱 정해서 뱉는 게 아닙니다. 대신 **정규분포(Gaussian Distribution)**를 만듭니다.

- 출력 1 (Mean, $\mu$): "내 생각엔 여기가 정답 같아." (가장 확률 높은 행동)

- 출력 2 (Std, $\sigma$): "근데 아닐 수도 있으니까, 이만큼은 좌우로 흔들어봐." (범위)실제로 로봇한테 내려가는 최종 명령은 아래 공식으로 만들어집니다.$$\text{Action} = \mu \text{ (뇌의 정답)} + \text{Noise} \times \sigma \text{ (std)}$$

2. 🎲 std의 역할: "탐험가 점수"

**std가 클 때 (초반):**

뇌: "다리를 0.5로 들어... 아니? 1.0? 아니 -0.5?"의미: "나 아직 잘 모르겠어! 이것저것 막 해볼래!"

현상: 로봇이 미친 듯이 다리를 텁니다(Shaking). 평균은 0.5여도 실제로는 0.1 갔다가 0.9 갔다가 난리가 나기 때문입니다.

목적: 우연히 좋은 동작을 발견하기 위함 (탐색, Exploration).

**std가 작을 때 (후반):**

뇌: "다리를 0.5로 들어. 확실해. 아주 조금만(0.01) 흔들어봐."의미: "나 이제 고수야. 내 정답(Mean)을 믿어."

현상: 로봇이 부드럽게 움직입니다. 노이즈가 거의 없어서 뇌가 의도한 대로 깔끔하게 움직입니다.

목적: 아는 대로 잘하기 위함 (활용, Exploitation).

**요약**

질문하신 대로 **std는 액션 명령에 섞이는 '노이즈(무작위성)의 크기'**가 맞습니다.

지금 std가 줄어들고 있다는 건, 로봇이 **"확신"**을 가지기 시작했다는 아주 좋은 신호입니다!


1. 🧮 일반적인 수식 (데이터가 N개 있을 때)우리가 배치(Batch)로 모은 데이터 $x_1, x_2, ..., x_N$이 있을 때의 표준편차 $\sigma$ (시그마)는 다음과 같습니다.
$$\sigma = \sqrt{\text{Variance}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$
$\sigma$: 표준편차 (Standard Deviation)

$N$: 데이터의 개수 (예: batch_size = 256)

$x_i$: 각 데이터 값 (예: 각 스텝에서의 보상 값)

$\mu$: 데이터들의 평균 ($\frac{1}{N}\sum x_i$)[의미]"평균($\mu$)으로부터 각 데이터가 평균적으로 얼마나 떨어져 있는가?"분산은 제곱을 해서 단위가 뻥튀기되지만(예: 점수$^2$), 표준편차는 다시 루트를 씌워서 원래 데이터와 단위가 같아집니다. (예: 점수)

2. 🤖 PPO 로그의 std (Gaussian Policy)사용자님의 로그에 찍히는 train/std는 위처럼 데이터를 다 모아서 계산하는 게 아니라, 로봇의 뇌(Policy)가 가지고 있는 "행동 범위 파라미터" 그 자체입니다.PPO는 행동(Action)을 **정규분포(Gaussian Distribution)**에서 뽑습니다. 그 확률 밀도 함수 수식에서 $\sigma$가 바로 표준편차입니다.

$$\pi(a|s) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( - \frac{(a - \mu)^2}{2\sigma^2} \right)$$

$\mu$ (평균): 로봇이 생각하는 "최적의 행동" (뉴럴 네트워크가 계산한 값

)$\sigma$ (표준편차): 로봇이 그 행동 주변을 "얼마나 탐험(Randomness)할지" (로그에 찍히는 std)

$\sigma$가 크면: 그래프가 납작하고 넓게 퍼짐 $\rightarrow$ "이거저거 다 해봐!" (탐험)

$\sigma$가 작으면: 그래프가 뾰족함 $\rightarrow$ "난 내 판단을 확신해, 딱 이것만 할 거야." (수렴)



## approx_kl

$$D_{KL}(\pi_{old} || \pi_{new}) = \mathbb{E} \left[ \log \frac{\pi_{old}(a|s)}{\pi_{new}(a|s)} \right] = \mathbb{E} [ \log \pi_{old}(a|s) - \log \pi_{new}(a|s) ]$$




$$\text{Explained Variance} = 1 - \frac{\text{Var}(y_{\text{true}} - y_{\text{pred}})}{\text{Var}(y_{\text{true}})}$$여기서 각 변수의 의미는 다음과 같습니다:$$
- $y_{\text{true}}$ (실제값): 로봇이 실제로 획득한 보상의 합 (Return).
- $y_{\text{pred}}$ (예측값): Critic이 "이만큼 받을 거야"라고 예측한 값 (Value).
- $y_{\text{true}} - y_{\text{pred}}$ (잔차, Residual): 예측이 빗나간 정도 (오차).
- $\text{Var}$ (분산): 데이터가 평균으로부터 얼마나 퍼져있는지 나타내는 값.

다만, 컴퓨터(PPO 알고리즘)가 실제로 계산할 때는 무한한 연속 함수를 적분하는 것이 아니라, 우리가 수집한 데이터 묶음(Batch)에 들어있는 숫자들의 리스트를 가지고 통계적인 분산을 구합니다. 강화학습에서 학습은 배치(Batch) 단위로 이루어집니다. 예를 들어 batch_size=256이라면, 로봇이 256번 움직이면서 얻은 실제 가치(Returns, $y_{\text{true}}$) 값 256개가 리스트에 들어있겠죠.

이 리스트를 $Y = \{y_1, y_2, ..., y_N\}$ 이라고 할 때, 식은 중학교 때 배운 "분산 공식" 그대로입니다.$$\text{Var}(Y) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2$$

평균 구하기 ($\bar{y}$): 256개 값의 평균을 먼저 구합니다.$\bar{y} = \frac{1}{N} \sum y_i$편차 제곱 (Difference Squared): 각 데이터가 평균에서 얼마나 떨어져 있는지 구해서 제곱합니다. (마이너스를 없애기 위해 제곱)$(y_1 - \bar{y})^2, (y_2 - \bar{y})^2, ...$평균 내기: 이 제곱한 값들을 다시 다 더해서 $N$으로 나눕니다.

---

강화학습(PPO)은 [실습] $\rightarrow$ [공부] $\rightarrow$ [성적표] 순서로 진행됩니다.

1단계: 실습 (Experience Collection)n_steps = 2048로봇이 세상에 나가서 2048발자국을 움직이며 데이터를 모읍니다.이때는 학습(가중치 업데이트)을 하지 않습니다. 그냥 데이터만 쌓습니다.중요: 이 단계가 끝날 때까지는 로그가 안 찍힙니다.

2단계: 공부 (Update / Training) -> 여기가 헷갈리신 부분!이제 모아온 2048개 데이터를 가지고 책상에 앉습니다.이 2048개를 한 번에 다 공부하기 힘드니까, **batch_size = 256**개씩 쪼갭니다.$2048 \div 256 = 8$덩어리(Mini-batch)가 나오죠?[핵심] PPO는 이 8덩어리를 한 번만 보고 버리는 게 아니라, 보통 10번 정도 반복해서 봅니다. (이걸 n_epochs라고 합니다. 기본값 10)즉, (8덩어리 업데이트) × 10번 반복 = 총 80번의 업데이트가 순식간에 일어납니다.

3단계: 성적표 (Logging)공부가 다 끝나면, 그때서야 비로소 **로그 테이블(아까 보신 그 표)**이 "짠!" 하고 출력됩니다.여기에 찍히는 값(explained_variance, loss 등)은 방금 공부한 2048개 데이터에 대한 최종 요약본입니다.

요약 : 총 2048스텝을 학습하는데 256개의 스텝(mini-batch, batch_size = 256)으로 나눠서 8번 학습하고, 이것을 10번 복습(n_step = 10), 총 80번의 업데이트 => 이것이 1 Iteration(Rollout, Epoch), 2048스텝을 모으고 학습하는 큰 사이클 한 번 -> 이것을 기준으로 Log Table이 출력됨