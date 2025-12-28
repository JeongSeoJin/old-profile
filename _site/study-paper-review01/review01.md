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
# Paper Review : "Design Principles for Energy Efficient Legged Locomotion and Implementation on the MIT Cheetah Robot"
---
## What is this paper regarding?
I reviewed this paper to gain insights into **actuator design** and **overall energy management** for legged locomotion to apply these principles to my robotic actuators. This paper represents **three major energy-loss**
1. **Heat loss from actuator**
2. **Friction loss in transmission**
3. **Interaction loss between system and environment** 

![figure2](figure2.png)
>$E_{loss} = E_j + E_f + E_i$

I was particularly impressed by the intuitive analysis of the **energy flow cycle** in respect of actuator. Also, systematic methods to minimize these losses are well organized, such as employing

- **High Torque Density Motor**
- **Energy Regenerative Electronic System**
- **Low Impedance Transmission(Back Drivability)**
- **Low Inertia Leg**

It ultimately uses the **Cost of Transport ($CoT = \frac{power}{weight \times velocity}$)** as a key metric to evaluate the total energy efficiency of the locomotive system.

<td><img src="figure1.png" width="400" alt="Actuator image 1" /></td>

---

## Key Contributions
- **Energy Analysis** : effectively reduced energy loss based on energy flow of the entire locomotive system. 
- **High Efficiency** : MIT Cheetah achieved a cost of transport of 0.5 that rivals running animals and significantly lower than other running robots of its time.
- **Proprioceptive Actuator Design** : Demonstrated the effectiveness of low-gear ratio actuators(QDD) for dynamic interactions

---

## Background of why I read this paper
In [my previous project](/project-c-qdd-actuator/c-qdd-actuator.md), **Development of Cycloidal-QDD Actuator**, I focused primarily on a mechanical implementation of my actuator to ensure the actuator worked. However, I realized the torque density and energy management system were not optimized. I lacked a comprehensive understanding of how individual actuator efficiency impacts the overall locomotive system. 

This paper was perfect resource to bridge that gap. It clearly organizes the principle of energy-efficient actuator in dynamic locomotion. As I'm now focusing on building high-performance actuators for successful dynamic control, understanding the principle applied to MIT Cheetah is crucial for my future work.

---

## Critical Thinking & Takeaway

#### 1. Energy Flow Diagram
![figure2](figure2.png)
Through above energy flow diagram in locomotion, I could analyze what exactly causes the energy loss in my actuator. 

#### 2. High Torque Density Motor(Large Gap Radius Motor) related to $E_j$
the electromagnetic(EM) motors with high-torque density minimize energy loss in actuators. Torque Density($\frac{torque}{mass}$) directly concerns the Joule heating of the EM motor by reducing the required electric current to provide torque. 

Continuous torque of the motor represents how much torque
the motor can generate at a constant heat dissipation.
The continuous torque is directly related to motor constant, which represents torque per square root of Joule heating.

$$K_M =  \frac{\tau}{\sqrt{I^2R}}$$

where:  
- $\tau : torque$  
- $I : current$  
- $R : resistance$ $$

For example, if the torque density of the motor doubles without changing other factors, the Joule heating Ej can be reduced by 75%.

This means $K_M$ doubles, since torue doubles assuming mass remains constant

$$\tau =  K_M \times \sqrt{I^2R}$$

Then to get a same torque, required current halved compared to the baseline. Therefore Joule Heating Loss($I^2R$) is a quarter of baseline which influences significant improved efficiency of the actuators 

<div style="overflow-x: auto;">
| Parameter | Baseline | Improved | Logic |
| :--- | :---: | :---: | :--- |
| **Torque Density** | $1$ | **$2$** | **Doubled** (Given condition) |
| **Motor Constant ($K_m$)** | $K$ | $2K$ | Torque capacity doubles while mass stays constant |
| **Required Current ($I$)**<br><small>(for same torque)</small> | $I_{req}$ | $\frac{1}{2} I_{req}$ | Current **is halved** due to doubled $K_m$ |
| **Joule Heating Loss ($E_j$)**<br><small></small> | $P_{loss}$ | $\frac{1}{4} P_{loss}$ | Heating scales with current squared<br>$(\frac{1}{2})^2 = \frac{1}{4}$ |
| **Reduction in Heat Loss** | - | **75% reduction** | $100\% - 25\% = \mathbf{75\%}$ **saved** |
{: .table .table-striped style="width: 100%; display: table; font-size: 0.9em;"}
</div>

Consequently, increasing the torque density of the EM motor is highly desirable for efficiency reducing Joule Heating Loss($E_j$)

Then, How can we get a high torque density EM motors? The torque density of the motor can be improved by increasing the gap radius: the radius of the gap between the motor stator
windings and the permanent magnets on the rotor.

The torque density scales by approximately $\frac{\tau}{m} \propto r_{gap}$, the torque per inertia scales by $\frac{\tau}{J} \propto r_{gap}^{-1}$, and the torque
squared per electric power, also known as motor constant
square, a measure of torque production efficiency, scales by $\tau^2/I^2R = K^2_M \propto r_{gap}^3$. the results will favor the motor with the larger gap radius because it will have a smaller gear ratio and fewer gear-train stages; this results in less friction loss, higher torque density, and higher bandwidth.
<td><img src="figure5.png" width="400" alt="Figure5" /></td>

#### 3. Low Impedance Mechanical Transmission(low gear ratio transmission) related to $E_f$
Employing gears significantly reduces the torque demands on the motors while increasing torque density and reducing Joule Heating Loss($E_j$). However, employing gears increase mechanical impedance like reflected inertia, reflected damping and gear friction while reducing back drivability, transmission transparency, efficiency. Futher more, it may limit the control of the robot dynamics.

To minimize the lossess($E_f$) associated with employment of gears, the low number of gear stages is ineviatble especially in planetary gear which accumulate backlash stage after stage. And relatively low ratio reduces the contribution of reflected actuator dynamics on the mechanical impedance of the transmission. Therefore, low gear ratio is essential for efficient actuator design and the control for the dynamic robots


#### 4. Low Inertia Leg
Low Inertia Leg is especially cruicial for dynamic locomotion, such as  higher stride frequency running increasing high bandwidth of the leg. Additionally Low Inertia leg mitigate impact loss($E_i$) at touch down at every step. A large portion of the kinetic energy of the leg should be dissipated not only in the cyclic motion but also when the legs collide with the ground. therefore, employing low mass, inertia to the dynamic robots is critical for reducing energy loss caused by interaction between robot and environment. 

---

## Conclusion
In conclusion, understanding of the energy flow diagram is a crucial step for designing actuators. Based on these insights, Next version of my robotic actuator for dynamic robots will be designed minimizing the energy-losses mechanism employing high torque densit y EM motor, low impedance in transmissions and low inertia leg for future dynamic locomotive robots. 

---
## Reference
**Citation**: Seok, Sangok et al. “Design Principles for Energy-Efficient Legged Locomotion andImplementation on the MIT Cheetah Robot.” IEEE/ASME Transactions on Mechatronics 20.3(2015): 1117–1129.

**Publication**: http://dx.doi.org/10.1109/TMECH.2014.2339013

**NOTE:**
And this article is available as a open access article in **DSpace@MIT**!