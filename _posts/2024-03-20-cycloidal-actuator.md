---
layout: post
title: "Development of High-Performance Cycloidal QDD Actuator"
date: 2024-03-20 12:00:00 +0900
categories: [Project]
tags: [Robotics, Actuator, QDD, Mechanical Design]
permalink: /quasi-direct-drive-actuator/
pin: true
---

## 1. Abstract
The goal of this project was to design and manufacture a **Quasi-Direct Drive (QDD) actuator** tailored for dynamic legged robots. Unlike traditional high-ratio gearboxes, QDD actuators prioritize **back-drivability** and **torque transparency**.

I developed a custom **15:1 cycloidal reducer** integrated with a high-torque BLDC motor. The system utilizes **Field Oriented Control (FOC)** via a Moteus controller to achieve precise torque control. This report details the design methodology, manufacturing process using 3D printing (PA-CF12) and CNC machining, and the validation of the system.

![Project Thumbnail or Final Render](/assets/img/project_thumbnail.jpg)
*(Replace this with your main project image)*

---

## 2. Motivation: Why Cycloidal QDD?
Dynamic robots, such as quadrupeds (e.g., MIT Cheetah), require actuators that can handle impact loads and provide proprioceptive feedback. Traditional gearboxes (harmonic drives) are fragile and expensive.

* **High Torque Density:** Cycloidal gears offer high reduction ratios in a compact space.
* **Impact Resistance:** The rolling contact mechanism distributes load across multiple pins, making it shock-resistant.
* **Cost-Effectiveness:** Designed to be manufacturable using accessible equipment like 3D printers and basic CNC.

---

## 3. System Architecture & Components

### 3.1 Mechanical Design
The core of the actuator is the cycloidal disk and the eccentric shaft. I designed a dual-disk configuration to cancel out vibrations.
* **Reduction Ratio:** 15:1
* **Module:** Custom cycloidal profile generated via Python script in Onshape.

### 3.2 Electromagnetic Design
To maximize torque output, I custom-wound the stator.
* **Configuration:** **36N42P** (36 Slots, 42 Poles).
* **Rotor:** 42 N52-grade Neodymium magnets bonded with high-strength epoxy.
* **Air Gap:** The rotor geometry was optimized to achieve a **minimal air gap of 0.5mm**. This tight clearance maximizes the **magnetic flux linkage**, significantly enhancing electromagnetic force and torque efficiency.

![Exploded View CAD](/assets/img/exploded_view.jpg)
*(Insert Exploded View CAD image here)*

### 3.3 Electronics & Control
For precise torque control and dynamic response, I integrated the **Moteus-c1 controller** (mjbots). This controller is widely adopted in the dynamic robotics community for its compact form factor and high bandwidth.

It implements **Field Oriented Control (FOC)**, essential for smooth torque generation. With a wide input range (10-51V) and **20A peak phase current**, it provides sufficient power capacity to drive the custom-wound 8110 stator to its full potential.

---

## 4. Assembly & Tolerance Strategy
Precise assembly is critical to minimize backlash. I implemented distinct **tolerance strategies** based on material properties.

### 4.1 Bearing Installation & Fits
For housing components printed in **PA-CF12 (Carbon Fiber Nylon)**, I designed a **0.1mm interference fit**. This accounts for the material's slight compliance and thermal shrinkage.

Conversely, for **CNC-machined Aluminum 6061** parts, a tighter **0.02mm interference fit** was applied due to the metal's rigidity. I reinforced these interfaces with a thin application of **JB Weld epoxy** to prevent micro-movements.

### 4.2 Output Mechanism
I installed six **M2x20mm stainless steel shafts** as carrier pins. These shafts were precision-aligned to allow the external rollers to **rotate freely**, minimizing friction at the output stage while maintaining structural rigidity.

![Assembly Process](/assets/img/assembly_photo.jpg)
*(Insert a photo of the real assembly here)*

---

## 5. Control & Validation
(Validation content... e.g., "I tested the actuator using a CAN-FD interface...")

---

## 6. Limitations & Future Works

### 6.1 Limitations of the Current Prototype
Through testing, two primary limitations were identified:

**1. Suboptimal Magnetic Flux Path**
The current rotor lacks a **ferromagnetic back-iron** or Halbach array. Consequently, magnetic flux is not effectively focused inward, leading to **lower torque density** than theoretically possible.

**2. Structural Rigidity of the Eccentric Shaft**
The eccentric input shaft is currently a **multi-part assembly** fastened with bolts. During operation, the **strong radial magnetic attraction** compromises the alignment, causing slight structural deflection and oscillation at high speeds.

### 6.2 Future Works
* **Monolithic Shaft:** Machine the eccentric shaft as a single piece to improve rigidity.
* **Steel Back-Iron:** Add a steel ring to the rotor to close the magnetic circuit and boost torque.

---

## 7. Credits
* **Hardware Design:** Seo Jin Jeong
* **Firmware:** Moteus (mjbots) Reference

## 8. References
1.  *Moteus Controller Documentation*, mjbots.
2.  *Design of Cycloidal Drives*, James G.