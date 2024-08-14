
## Optimization

This example's plots are generated using [example_optimization.py](../example_optimization.py).

We choose the loss function as the 2d-rosenbrock function, with a circle constraint. 
The parallel optimization algorithm used in this case is [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) via [``nevergrad``](https://github.com/facebookresearch/nevergrad)
(other parallel optimization options are [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) via [``nevergrad``](https://github.com/facebookresearch/nevergrad) 
or [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) via [``botorch``](https://github.com/pytorch/botorch)).

We pick 10 workers times 30 iterations. Progression of the loss with the number of iterations:
![example_optimization_loss_progress](pics/example_optimization_loss_progress.png)

2d visualization of the loss function (log absolute of the values), 
the circle constraint (white line), 
and the points approaching (dark to bright) the minima (marked by a star):
![example_optimization_2d_visualization](pics/example_optimization_2d_visualization.png)

In the example above the number of iterations was arbitrary, but for model calibration the loss function should be a 
measure of the model's fit to data, and the optimization goal is to reach a statistically satisfactory value before 
proceeding to the Bayesian analysis.
