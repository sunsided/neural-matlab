# Neural Networks in MATLAB

Experiments with Neural Networks in MATLAB.

```matlab
a = input;

for j=1:numel(layers)  
    weights =  layers{j}.theta;
    sigma   =  layers{j}.sigma;
    
    z = weights * [1; a];
    a = sigma(z);
end

output = a;
```

Start with `ann.m`. This code implements a training example and utilizes the following functions:

* `feedforward.m`: Runs inputs through the neural network, producing the hypothesis of the result.
* `feedforward_for_training.m`: Like `feedforward.m`, but produces additional output required for the backpropagation stage.
* `backpropagate.m`: Performs the error backpropagation and produces weight deltas.

Depending on the configuration, one of the following Gradient Descent algorithms can be used:

* `momentum_gradient_descent.m`: This is regular gradient descent with a fixed learning rate that additionally uses a momentum term to add in fractions of the previous gradient.
* `accelerated_gradient_descent.m`: This implementation utilizes adaptive (i.e. different) delta values per weight parameter instead of a single learning rate and only uses the gradient to check for direction changes.

In all cases, the concept of *flat spot elimination* (FSE) can be used to aid in navigating areas of small error gradients by artificially providing an empirically set gradient direction to move along. Small values such as `0.1` might work, disable by setting it to `0`. 