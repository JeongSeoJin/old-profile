
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
- **Low Impedance Transmission(Back Drivability**
- **Low Inertia Leg**

It ultimately uses the **Cost of Transport ($CoT = \frac{power}{weight \times velocity}$)** as a key metric to evaluate the total energy efficiency of the locomotive system.

![figure1](figure1.png)

## Key Contributions
- **Energy Analysis** : effectively reduced energy loss based on energy flow of the entire locomotive system. 
- **High Efficiency** : MIT Cheetah achieved a cost of transport of 0.5 that rivals running animals and significantly lower than other running robots of its time.
- **Proprioceptive Actuator Design** : Demonstrated the effectiveness of low-gear ratio actuators(QDD) for dynamic interactions


## Background of why I read this paper
In [my previous project](/project-c-qdd-actuator/c-qdd-actuator.md), **Development of Cycloidal-QDD Actuator**, I focused primarily on a mechanical implementation of my actuator to ensure the actuator worked. However, I realized the torque density and energy management system were not optimized. I lacked a comprehensive understanding of how individual actuator efficiency impacts the overall locomotive system. 

This paper was perfect resource to bridge that gap. It clearly organizes the principle of energy-efficient actuator in dynamic locomotion. As I'm now focusing on building high-performance actuators for successful dynamic control, understanding the principle applied to MIT Cheetah is crucial for my future work.

## Critical Thinking & Takeaway

### Energy Flow Diagram
![figure2](figure2.png)
Through above energy flow diagram in locomotion, I could analyze what exactly causes the energy loss in my actuator. 

### High Torque Density Motor(Large Gap Radius Motor)
the electromagnetic(EM) motors with high-torque density minimize energy loss in actuators. Torque Density($\frac{torque}{mass}$) directly concerns the Joule heating of the EM motor by reducing the required electric current to provide torque. 

Continuous torque of the motor represents how much torque
the motor can generate at a constant heat dissipation.
The continuous torque is directly related to motor constant, which represents torque per square root of Joule heating.

$$K_M =  \frac{\tau}{\sqrt{I^2R}}$$

where:  
- $\tau = torque$  
- $I = current$  
- $R = resistance$ $$

For example, if the torque density of the motor doubles without changing other factors, the Joule heating Ej can be reduced by 75%.

This means $K_M$ doubles, since torue doubles assuming mass remains constant

$$\tau =  K_M \times \sqrt{I^2R}$$

Then to get a same torque, required current halved compared to the baseline. Therefore Joule Heating Loss($I^2R$) is a quarter of baseline which influences significant improved efficiency of the actuators 

| Parameter | Baseline | Improved | Logic |
| :--- | :---: | :---: | :--- |
| **Torque Density** | $1$ | **$2$** | **Doubled** (Given condition) |
| **Motor Constant ($K_m$)** | $K$ | $2K$ | Torque capacity doubles while mass stays constant |
| **Required Current ($I$)**<br><small>(for same torque)</small> | $I_{req}$ | $\frac{1}{2} I_{req}$ | Current **is halved** due to doubled $K_m$ |
| **Joule Heating Loss ($E_j$)**<br><small></small> | $P_{loss}$ | $\frac{1}{4} P_{loss}$ | Heating scales with current squared<br>$(\frac{1}{2})^2 = \frac{1}{4}$ |
| **Reduction in Heat Loss** | - | **75% reduction** | $100\% - 25\% = \mathbf{75\%}$ **saved** |

Consequently, increasing the torque density of the EM motor is highly desirable for efficiency.

So How can we get a high torque density EM motors?

### Energy Regeration

### Low Impedance Mechanical Transmission(low gear ratio transmission)


![figure5](figure5.png)


