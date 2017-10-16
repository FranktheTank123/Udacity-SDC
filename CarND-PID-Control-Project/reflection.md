# PID Controller Reflection

## Q: Describe the effect each of the P, I, D components had in your implementation.

### P - Proportional

The proportional term computes an output proportional to the cross-track error (cte). A imple P-controller is unstable and at best oscillates about the setpoint.

### D - Differential

The oscillations caused by purely D control can be mitigated by a term proportional to the derivative of the cross-track error. The derivative gain contributes a control output of the form -K_d d/dt cte, with a positive constant K_d.

### I - Integral

A third contribution is given by the integral gain which simply sums up the cross-track error over time. The corresponding contribution to the steering angle is given by -K_i sum(cte). Thereby, biases can be mitigated, for instance if a zero steering angle does not correspond to a straight trajectory. At high speeds this term can also be useful to accumulate a large error signal quickly, for instance when the car is carried out sideways from the reference trajectory. This allows to reduce proportional gain, which causes oscillations at high speeds. It is also beneficial to limit the memory of this term to avoid overshooting. Here, we used an exponentially weighted moving average for this purpose.


## Q: Describe how the final hyperparameters were chosen.

All hyperparameters were chosen manually (which is very similar to Twiddle). The reason for choosing this method is that

1. it is very error sensitive. Tiny deviation from the optimal parameters could quickly make the car going out of the track.
2. There seems no easy way to initialize multiple runs of the simulation (i.e., human efforts are required for each restart). Can someone introduce an API to restart the simualor?

Here are the huristic of tuning the parameter:

1. First, only tune `P` with `K_i=K_d = 0`
2. Then incresae `K_d` incrementally until the oscillations reduces.
3. Increase `K_i` slightly until it runs longer/stabler.
4. If crashed:
    * if caused by slow reaction: increase `K_p` or `K_i`
    * if caused by oscillations: reduce `K_p` or `K_i`



