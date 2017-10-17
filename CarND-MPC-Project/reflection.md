# MPC Reflection

## The Model
_Requirement: Student describes their model in detail. This includes the state, actuators and update equations._

Here we refer to the class materials:

### State = [x, y, psi, v]

* **x**: x position
* **y**: y position
* **psi**: vehicle orientation
* **v**: velocity

### Actuators = [delta, a]

* **delta**: steering angle
* **a**: acceleration, in this case throttle

### Update Equation

* x = x + v * cos(psi) * dt
* y = y + v * sin(psi) * dt
* v = v + a * dt
* psi = psi + (v / L_f) * delta *dt

### MPC Algo

1. Receive (or initialize) data and update the current state
2. Use **Solver**
    * input: current state
    * output: control inputs that minimize the cost function
3. Apply the **1st** control output from (2) into the vehicle
4. Repeat.

## Timestep Length and Elapsed Duration (N & dt)
_Requirement: Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried._

 `dt` cannot be too low (e.g., 0.05), which will cause the vehicle to drive wiggyly. We want the reaction (waypoints) close to human reaction time, which is around 0.1. When `dt` is too high (e.g., 0.2), we see the vehicle response a bit slower to the curvature. **Thus we chose `dt=0.1`**.

`N` cannot be too small, as we won't be able to capture (i.e., fit the polynomial correctly) the curvature with smaller N. `N` cannot be too big neither. I tried `N = [10, 15, 20, 25, 30, 40, 50]` and `N=15` seems reasonable.


## Model Predictive Control with Latency
_Requirement: The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency._

It's important to incorporate the latency in the control. Otherwise, the car will always response 100ms behind the reality, causing unstable behaviors like overturning and underturning. In order to not let this happen, we use the kinematic model to predict what will happen to the road 100ms in the future because passing the states into the MPC solver.