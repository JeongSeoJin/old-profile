# actuator network를 학습할 때 가상 데이터의 실효성
강화학습에서 대개 sim2real gap(discrepancies b/w reality and simulation)을 줄일 수 있는 주된 요소는 Improved Simulation 및 Robust Policy가 있다. 
- Improved Simulation
    - system identification
    - accurate actuator modeling
    - simulating latency

- Robust Policy
    - randomizing the physical environment
    - adding perturbations
    - designing a compact observation space

Reallity Gap 을 극복하기 위해서는 특히 두 개의 가장 큰 문제를 해결해야한다. 

1. inaccurate actuator modeling
2. lack of latency

아무리 policy에 perturbances & randomization으로 robust하게 만들어도 actuator이 너무 큰 비선형적인 방해요소들을 갖고 unpredictable하게 동작하면 모델을 결국 무너질 수 밖에 없다. 
때문에 이러한 문제를 해결하기 위해서 actuator 자체에 network를 심어서 input 대해 output이 예측가능하도록 네트워크를 학습한다. 

---

두 가지 차원의 randomization
- Environment randomization : 로봇의 무게, 링크의 길이, 질량중심, 외부에서 미는 힘, 경사도 등
    - Simulator 상에서 parameters를 randomize
    - 외부의 예기치 못한 변수를 대응할 수 있음

- **Actuator Randomization** : 모터 내부의 온도, 전압 강하, 통신 지연, 기어 백래시 등
    - Actuator Network(MLP)를 통해서 이상적인 모터 수식을 현실적인 모터 모델로 만들어줌
    - 로봇의 뇌는 학습할 때부터 '아 내가 10만큼의 힘을 줘도 실제로는 8밖에 안 나가고 0.02초 늦게 반응하는구나'라고 직접 느끼면서 학습한다


**장점 : "현실에서 얻을 수 없는 데이터"를 만들 수 있습니다. (Safety & Cost)**
실제 로봇으로 학습하려면 데이터를 모아야 하는데, 이게 정말 힘듭니다.

위험한 상황: 로봇 팔이 최고 속도로 움직이다가 벽에 부딪히는 상황, 배터리가 과열돼서 전압이 뚝 떨어지는 상황 등을 실제로 실험하면 로봇이 부서집니다.

비용과 시간: 데이터 100만 개를 모으려면 로봇을 몇 달 동안 돌려야 합니다. 모터가 마모되고, 기어가 갈립니다.

**가상 데이터(Synthetic Data)**를 쓰면?

로봇을 부수지 않고도 **"극한 상황(Extreme Cases)"**을 무제한으로 만들어 학습시킬 수 있습니다.

모델은 평소에는 얌전하다가, 갑자기 센서가 튀거나 충격이 가해져도 "아, 나 이거 시뮬레이션에서 봤어" 하고 대처할 수 있게 됩니다.


**"데이터의 순도" (Clean vs Dirty)**

실제 센서 데이터는 "왜 이렇게 됐는지" 이유를 알려주지 않습니다. 그냥 결과값만 띡 던져주죠. (노이즈인지, 진짜 충격인지 구분 불가)

하지만 가상 데이터는 우리가 **Ground Truth(정답)**를 완벽하게 알고 있습니다.

"지금 튀는 값은 내가 노이즈를 섞어서 그런 거야."

"지금 지연이 생긴 건 인덕턴스 때문이야."

이렇게 원인과 결과를 명확히 아는 데이터로 학습시키면, 모델이 인과관계를 훨씬 더 정확하고 빠르게 배웁니다.