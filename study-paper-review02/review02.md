<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      displayMath: [['$$','$$'], ['\\[','\\]']],
      processEscapes: true
    }
  });
</script>
---
# Paper Review : "Demonstrating Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot"
---
## What is this paper regarding?
This paper demonstrates accessible 3d-printed low-cost humanoid robot promoting democratization in humanoid robotics. The core design is a modular 3d-printed customizable actuator easily replaceable, preventing delays of an experiment from broken parts of robots. Also conduct experiments to validate reliability of 3d-printed actuator. Futhermore, It showcased zero-shot policy transfer from simulation to hardware, highlighting the platform's suitability for research validation.

  <table>
    <tr>
      <td><img src="fig1.png" width="250" alt="Actuator image 1" /></td>
      <!-- <td><img src="fig2.png" width="300" alt="Actuator image 2" /></td> -->
    </tr>
  </table>

---
## Key Contributions
- Low Cost Accessible Modular 3d-printed actuator
- Actuator Evaluation of 3d-printed actuator
- Zero-shot policy transfer from simulation to hardware

---
## Background of why I read this paper
Since I am planning to build a humanoid robot in the near future, the system must be reliable enough for successful Reinforcement Learning (RL) locomotion experiments, even if it is based on low-cost 3D-printed parts. Also, components should be quickly replaceable to accelerate the experimental cycle.

I'm now focusing on building reliable robotic actuator to employ reinforcement learning successfully. In my prvious project, I did not conduct evaluations. Thus this paper is highly suitable for me, offering insights not only into actuator evaluation methods but also deploying reinforcement learning model on my robots.

---
## Critical Thinking & Takeaway
#### 1. Accessible Low-Cost Actuator
The actuators are two different sizes(6512, 5010). As shown in the figures below, It's relatively low cost compared to other metallic actuators. For example, 6512 actuator uses M6C12 150KV BLDC drone motor from MAD Components, AS5600 encoder and B-G431B-ESC1 as a motor driver. The reducer, and housing parts are all 3d printed with PLA. The cycloidal gears were selected as reducer due to its high robustness. 

<table>
  <caption>BOM & B-G431B-ESC1 Controller</caption>
  <tr>
    <td><img src="BOM.png" width="250" alt="Actuator image 1" /></td>
    <td><img src="controller.png" width="300" alt="Actuator image 2" /></td>
  </tr>
</table>

<table>
  <caption>as5600 Encoder & M6C12 150KV Motor</caption>
  <tr>
    <td><img src="encoder.png" width="250" alt="Actuator image 1" /></td>
    <td><img src="motor.png" width="300" alt="Actuator image 2" /></td>
  </tr>
</table>

<caption>Specification of 'M6C12 150kv BLDC Drone Motor'</caption>
<td><img src="specification.png" width="600" alt="Actuator image 2" /></td>

As you can see in the motor specification figure,
- Motor Size : D:72 x 27mm
- Number of the poles : 14
- RPM/V : 150KV
- Maximum Current : 36.18A
- Maximum Torque : 2.64Nm
- Nominal Battery : 12S lipo($\approx$ 48V) 

#### 2 Actuator Evaluation - Efficiency
A reliable actuator is fundamental to the robot's overall performance. This paper conduct a set of experiments under conditions identical to those on the robot, including 24V power supply, indentical position PD gains, and matching position torque and current bandwidth 
configurations.
<td><img src="fig7.png" width="600" alt="Actuator image 2" /></td>

In the experiment, Actuator Under Test performs Torque(Torque Control) and Damping Actuator performs constant speed(speed control). Two load cells measure torque of the primary actuator. An electrical-power measurement board logged the supply voltage and current, from which electrical input power was calculated.

$$P_M = \tau_m \times \omega_m$$

where:
- $P_M$ : Mechanical Output Power
- $\tau_m$ : Measured Torque 
- $\tau_m$ : Measured Rotational Speed

Through these testset, Mechanical Output Power can be calculated which is power of the actuator.

$$M_{eff} = \frac{P_M}{\tau_c \times v_c} $$

where:
- $M_{eff}$ : Mechanical Efficiency
- $\tau_{c}$ : Commanded Torque
- $v_c$ : Commanded Velocity

Then **Mechanical Efficiency** can be calculated either which represents the **efficiency of 3d-printed reducer**.

$$T_{eff} = \frac{P_{measured}}{P_{input}} $$

where:
- $T_{eff}$ : Total Efficiency
- $P_{measured}$ : Measured Mechanical Power
- $P_{input}$ : Input Power

Also, **Total efficiency** reflects the overall actuator
efficiency, which includes **motor copper losses, driver electrical losses, and mechanical losses.**

<td><img src="fig8.png" width="600" alt="Actuator image 2" /></td>

The experiment conducted the actuator performance across three different speeds. Each torque and speed command was maintained for one second. As shown in Figure 8, the gearbox exhibits a mechanical efficiency of approximately 90% across most operating conditions. However, at high torque and velocity, efficiency decreases due to heat generation.

This dynamometer method is suitable for rigorous evaluation separating mechanical efficiency and total efficiency. we can easily analyze contributions of mechanical losses and electrical losses. This is an effective method to evaluate custom actuators and reducers.


#### 3. Actuator Evaluation - Transmission stiffness
<td><img src="fig9.png" width="600" alt="Actuator image 2" /></td>

The actuator’s transmission stiffness is measured by rigidly fixing the output shaft relative to the actuator housing and measuring motor displacement under a range of static torques. 

The torque command was gradually ramped from 0 Nm to 20 Nm and back in both directions. A linear fit was then applied to the data collected from 4 Nm to 10 Nm, and the inverse of the slope yielded a stiffness of approximately 319.49 Nm/rad (Figure 9). The transmission stiffness can be enhanced by opting alternative materials(e.g PA-CF).

#### 4. Actuator - durability 
Durability is a primary consideration of 3d-printed actuator. The experiment was conducted for 60-hours lifting a pendulum(0.5kg, 0.5m) through range of -45 degrees to 90 degrees at a frequency of 0.5Hz. Consequently, efficiency and backlash remained within acceptable limits throughout the 60-hours test. The durability is remarkable especially considering cycloidal gear thickness is only 4mm. This finding gives me confidence in my own project, as my actuator design shares similar gear dimensions.

<td><img src="fig10.png" width="600" alt="Actuator image 2" /></td>

#### 5. Legged Locomotion
This paper achieve a direct sim-to-real transfer of a Isaac Gym-trained policy to the physical robot without relying on additional state estimation methods. This is crucial for locomotive robot using 3d printed actuator which causes backlash. The authors formulate the locomotion task as a Partially Observed Markov Decision Process(POMDP) and use a standard Proximal Policy Optimization(PPO) algorithm to learn a control policy.

proprioceptive observations from the robot hardware:

- the base angular velocity 
- the projected gravity vector 
- joint positions and velocites

Additional inputs include the commanded linear volocity provided by the user and the previous time-step action. Notably, the experiement utilized only 30% of the actuator's torque limit, suggesting that the actuators are operating well within their capacity. 

#### 6. Additional Insights
Based on the specifications of the 6512 motor (2.64 Nm) and a gear ratio of 15:1 with an estimated efficiency of 90%, the theoretical maximum torque is approximately 35.64 Nm.

However, in this experimental setup, I estimate the practical maximum torque to be restricted to around 20–24 Nm. While the input voltage is 24V (compared to the motor's rated 48V), the primary limiting factor is likely a software-imposed current limit designed to protect the 3D-printed plastic gears from structural failure, rather than the motor's raw capability.

---
## Conclusion
This paper validates that the low-cost 3d-printed actuator with zero-shot transfer policy is suitable for humanoid robot platform, evaluating actuator's reliability. Especially the overall building/evaluation process of humanoid robot is remarkable, including actuator designs/evaluations and Reinforcement Learning deployment. Furthermore, '[Berkeley Humanoid Lite Docs](https://berkeley-humanoid-lite.gitbook.io/docs)' deals with the specific contents of building this humanoid robot. Building the full humanoid robot with 3d-printed actuator, deploying the RL model on the robot, make me encourage and confident alot.  

---
## Reference
**Citation**: https://arxiv.org/pdf/2504.17249

**NOTE:** This paper is available as an open-access paper

Berkeley Humanoid Lite Docs : [https://berkeley-humanoid-lite.gitbook.io/docs](https://berkeley-humanoid-lite.gitbook.io/docs)

---
## Image Sources

[Controller image](https://www.st.com/en/evaluation-tools/b-g431b-esc1.html)

[Motor and Motor specification image](https://store.mad-motor.com/products/mad-m6c12-eee-brushless-motor-for-the-long-flight-time-multirotor-hexacopter-octopter?srsltid=AfmBOopSIaIIfPpFTQoUzMsGxo3gZUvFAoL8lv4r6elfjP7InwBlftJ5)

[Encoder image](https://www.tinytronics.nl/en/sensors/magnetic-field/as5600-magnetic-angle-sensor-encoder-module)