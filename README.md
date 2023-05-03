# MoMo
Pytorch implementation of MoMo methods. Adaptive learning rates for SGD with momentum (SGD-M) and Adam. 

``` python
from momo import Momo
opt = Momo(model.parameters(), lr=1)
```
or

``` python
from momo import MomoAdam
opt = Momo(model.parameters(), lr=1e-2)
```


## Recommendations

In general, if you expect SGD-M to work well on your task, then use Momo. If you expect Adam to work well on your problem, then use MomoAdam.

* The option `lr` and `weight_decay` are the same as in standard optimizers. As Momo and MomoAdam automatically adapt the learning rate, you should get good preformance without heavy tuning of `lr` and setting a schedule. Setting `lr` constant should work fine. For Momo, our experiments work well with `lr=1`, for MomoAdam `lr=1e-2` (or slightly smaller) should work well.

**One of the main goals of Momo optimizers is to reduce the tuning effort for the learning rate and get good performance for a wide range of learning rates.**

* For Momo, the argument `beta` refers to the momentum parameter. The default is `beta=0.9`. For MomoAdam, `(beta1,beta2)` have the same role as in Adam.

* The option `lb` refers to a lower bound of your loss function. In many cases, `lb=0` will be a good enough estimate. If your loss converges to a large positive number (and you roughly know the value), then set `lb` to this value (or slightly smaller).


