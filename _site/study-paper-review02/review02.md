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
      <td><img src="fig2.png" width="300" alt="Actuator image 2" /></td>
    </tr>
  </table>

## Key Contributions
- Low Cost Accessible Modular 3d-printed actuator
- Reliable Actuator Evaluation
- Zero-shot policy transfer from simulation to hardware

## Background of why I read this paper
This paper deals with 'low-cost', '3d-printed', 'modular actuator', 'zero-shot policy'. These properties are crucial for me planning to build humanoid robot in the near future. The robot has to be reliable for successful experiment of Reinforcement Learning(RL) in locomotion even it's based on low-cost 3d-printed actuator. Also It should be replaceable quickly to accelerate experiments.

I'm now focusing on building reliable robotic actuator to employ reinforcement learning successfully. In my prvious project, I have not conducted evaluation yet. Thus this paper is suitable for me, offering insights not only into actuator evaluation methods but also deploying reinforcement learning model on my assembled robots in future.


## Critical Thinking & Takeaway
#### Accessible Low-Cost Actuator
The actuators are two different size(6512, 5010). As you can see below figures, It's relatively low cost compared to other metalic actuators. For example, 6512 actuator uses M6C12 150KV BLDC drone motor from MAD Components, as5600 encoder and B-G431B-ESC1 as a motor driver. The reducer, and housing parts are all 3d printed with PLA. The cycloidal gears are opted as reducer owing to it's high robustness. 

<table>
  <caption>BOM and Controller Components</caption>
  <tr>
    <td><img src="BOM.png" width="250" alt="Actuator image 1" /></td>
    <td><img src="controller.png" width="300" alt="Actuator image 2" /></td>
  </tr>
</table>

<table>
  <caption>Encoder and Motor Components</caption>
  <tr>
    <td><img src="encoder.png" width="250" alt="Actuator image 1" /></td>
    <td><img src="motor.png" width="300" alt="Actuator image 2" /></td>
  </tr>
</table>

<caption>M6C12 150kv BLDC Drone Motor Specification</caption>
<td><img src="specification.png" width="600" alt="Actuator image 2" /></td>

As you can see in the motor specification figure,
- Motor Size : D:72 x 27mm
- Number of the poles : 14
- RPM/V : 150KV
- Maximum Current : 36.18A
- Maximum Torque : 2.64Nm
- Nominal Battery : 12S lipo($\approx$ 48V) 

#### Actuator Evaluation
A reliable actuator is fundamental to the robot's overall performance. This paper conduct a set of experiments under conditions identical to those on the robot, including 24V power supply, indentical position PD gains, and matching positionm torque and current bandwidth configurations.

## Conclusion

---
## Reference
**Citation**: 

**Publication**: 

**NOTE:**

[Controller](https://www.st.com/en/evaluation-tools/b-g431b-esc1.html)

[Motor and Motor specification](https://store.mad-motor.com/products/mad-m6c12-eee-brushless-motor-for-the-long-flight-time-multirotor-hexacopter-octopter?srsltid=AfmBOopSIaIIfPpFTQoUzMsGxo3gZUvFAoL8lv4r6elfjP7InwBlftJ5)

[Encder](https://www.tinytronics.nl/en/sensors/magnetic-field/as5600-magnetic-angle-sensor-encoder-module)