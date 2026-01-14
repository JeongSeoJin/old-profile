물리 (Hardware):
- Kd (Damping) 올리기 $\rightarrow$ 착지 시 팅김 방지 (스펀지 역할).
- action_scale 적용 $\rightarrow$ 관절 가동 범위 확보 (시원시원한 걸음).

보상 (Reward Engineering):
- 패널티는 작게 유지 (tracking_reward를 압도하지 않게).
- 대신 action_rate_penalty나 Action Filter를 써서 급격한 변화만 막기.

알고리즘 (Brain):
- ent_coef = 0.0: "이제 떨지 말고(std 낮추고) 확신을 가지고 걸어라."