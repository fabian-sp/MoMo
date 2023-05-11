# MoMo
Pytorch implementation of MoMo methods. Adaptive learning rates for SGD with momentum (SGD-M) and Adam. 

## Installation

You can install the package with

```
pip install momo-opt
```

## Usage

Import the optimizers in Python with

``` python
from momo import Momo
opt = Momo(model.parameters(), lr=1)
```
or

``` python
from momo import MomoAdam
opt = MomoAdam(model.parameters(), lr=1e-2)
```

**Note that Momo needs access to the value of the batch loss. You need to pass a ``closure`` into the ``.step()`` method that computes gradients and returns the loss.** 

``` python
def compute_loss(output, labels):
  loss = criterion(output, labels)
  loss.backward()
  return loss

# in each training step, use:
closure = lambda: compute_loss(output,labels)
opt.step(closure)
```
**For more details, see [a full example script](example.py).**




## Examples

### ResNet110 for CIFAR100

<p float="left">
    <img src="png/cifar100_resnet110.png" width="320" />
    <img src="png/cifar100_resnet110_training.png" width="305" />
</p>

### ResNet20 for CIFAR10


<p float="left">
    <img src="png/cifar10_resnet20.png" width="320" />
    <img src="png/cifar10_resnet20_training.png" width="305" />
</p>


## Recommendations

In general, if you expect SGD-M to work well on your task, then use Momo. If you expect Adam to work well on your problem, then use MomoAdam.

* The option `lr` and `weight_decay` are the same as in standard optimizers. As Momo and MomoAdam automatically adapt the learning rate, you should get good preformance without heavy tuning of `lr` and setting a schedule. Setting `lr` constant should work fine. For Momo, our experiments work well with `lr=1`, for MomoAdam `lr=1e-2` (or slightly smaller) should work well.

**One of the main goals of Momo optimizers is to reduce the tuning effort for the learning-rate schedule and get good performance for a wide range of learning rates.**

* For Momo, the argument `beta` refers to the momentum parameter. The default is `beta=0.9`. For MomoAdam, `(beta1,beta2)` have the same role as in Adam.

* The option `lb` refers to a lower bound of your loss function. In many cases, `lb=0` will be a good enough estimate. If your loss converges to a large positive number (and you roughly know the value), then set `lb` to this value (or slightly smaller). 

* If you can not estimate a lower bound before training, use the option `use_fstar=True`. This will activate an online estimation of the lower bound.


