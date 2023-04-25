"""
Adapted from:
	Novik, Mykola: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/types.py
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]