
## Optimization

This example's plots are generated using [example_optimization.py](../example_optimization.py).

We choose the loss function as the 2d-rosenbrock function, with a circle constraint. 

The parallel optimization algorithms available to use are:
1. [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) via [``nevergrad``](https://github.com/facebookresearch/nevergrad)
2. [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) via [``nevergrad``](https://github.com/facebookresearch/nevergrad) 
3. [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) via [``botorch``](https://github.com/pytorch/botorch)

We will use the first option here, and use 10 workers for 30 iterations. 

Progression of the loss with the number of iterations:
<img src="pics/example_optimization_loss_progress.png" alt="example_optimization_loss_progress" width="700" height="auto">

2d visualization of the loss function (log absolute of the values), 
the circle constraint (white line), 
and the points approaching (dark to bright) the minima (marked by a star):

<img src="pics/example_optimization_2d_visualization.png" alt="example_optimization_2d_visualization" width="700" height="auto">

In the example above the number of iterations was arbitrary, but for model calibration the loss function should be a 
measure of the model's fit to data, and the optimization goal is to reach a statistically satisfactory value before 
proceeding to the Bayesian model calibration (MCMC).
