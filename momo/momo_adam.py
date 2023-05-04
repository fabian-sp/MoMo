"""
Implements the MoMo-Adam algorithm.

Authors: Fabian Schaipp, Ruben Ohana, Michael Eickenberg, Aaron Defazio, Robert Gower
"""
import math
import warnings
import torch
import torch.optim
from .types import Params, LossClosure, OptFloat

class MomoAdam(torch.optim.Optimizer):
    def __init__(self, 
                params: Params, 
                lr: float=1e-2, 
                betas:tuple=(0.9, 0.999), 
                eps:float=1e-8,
                weight_decay:float=0,
                lb: float=0,
                divide: bool=True,
                use_fstar: bool=False):
        """
        Momo-Adam optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1e-2.
        betas : tuple, optional
            Momentum parameters for running avergaes and its square. By default (0.9, 0.999).
        eps : float, optional
            Term added to the denominator of Dk to improve numerical stability, by default 1e-8.
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        lb : float, optional
            Lower bound for loss. Zero is often a good guess.
            If no good estimate for the minimal loss value is available, you can set use_fstar=True.
            By default 0.
        divide : bool, optional
            Whether to do proximal update (divide=True) or the AdamW approximation (divide=False), by default True.
        use_fstar : bool, optional
            Whether to use online estimation of loss lower bound. 
            Can be used if no good estimate is available, by default False.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay
                        )
        
        super().__init__(params, defaults)

        self.lb = lb
        self.divide = divide 
        self.use_fstar = use_fstar
        
        # initialize
        self._number_steps = 0
        self.loss_avg = 0.
        self.state['step_size_list'] = list() # for storing the adaptive step size term

        return

    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.

        Returns
        -------
        (Stochastic) Loss function value.
        """
        
        with torch.enable_grad():
            loss = closure()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        _dot = 0. # = <d_k,x_k>
        _gamma = 0. # = gamma_k
        _grad_norm = 0. # = ||d_k||^2_{D_k^-1}
        
        self._number_steps += 1
        
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
  
            bias_correction1 = 1 - beta1 ** self._number_steps
            bias_correction2 = 1 - beta2 ** self._number_steps
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data           
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of inner product <grad, weight>
                    state['grad_dot_w'] = torch.tensor(0.).to(p.device)
                

                state['step'] += 1 
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']
                grad_dot_w = state['grad_dot_w']

                # Adam EMA updates
                grad_avg.mul_(beta1).add_(grad, alpha=1-beta1) # = d_k
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta1)

                bias_correction2 = 1 - beta2 ** state['step']
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps) # = D_k
                
                _dot += torch.sum(torch.mul(p.data, grad_avg))
                _gamma += grad_dot_w
                _grad_norm += torch.sum(grad_avg.mul(grad_avg.div(Dk)))
                    
        # Exponential moving average of function value
        # Uses beta1 of last param_group! 
        self.loss_avg = (1-beta1)*loss.detach() +  beta1*self.loss_avg 
        

        #################   
        # Update
        for group in self.param_groups:
            
            ### Compute adaptive step size
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            bias_correction1 = 1 - beta1 ** self._number_steps
            bias_correction2 = 1 - beta2 ** self._number_steps

            if self.use_fstar:  
                cap = ((1+lr*lmbda)*self.loss_avg + _dot - (1+lr*lmbda)*_gamma).item()         
                # Reset
                if cap < (1+lr*lmbda)*bias_correction1*self.lb:
                    self.lb = cap/(2*(1+lr*lmbda)*bias_correction1) 
                    self.lb = max(self.lb, 0) # safeguard 
                    
            nom = (1+lr*lmbda)*(self.loss_avg - bias_correction1*self.lb) + _dot - (1+lr*lmbda)*_gamma
                
            t1 = (max(nom, 0.)/_grad_norm).item()
            tau = min(lr/bias_correction1, t1)
            
            ### Update lb estimator
            if self.use_fstar:
                h = (self.loss_avg  + _dot -  _gamma).item()
                self.lb = ((h - (1/2)*tau*_grad_norm)/bias_correction1).item() 
                self.lb = max(self.lb, 0) # safeguard
                
            ### Update params
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']

                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)
                
                # AdamW-Pytorch way of weight decay
                if lmbda > 0 and not self.divide:
                    p.data.mul_(1-lmbda*lr)

                # Gradient step
                p.data.addcdiv_(grad_avg, Dk, value=-tau) # x_k - tau*(d_k/D_k)

                # Proximal way of weight decay
                if lmbda > 0 and self.divide:
                    p.data.div_(1+lmbda*lr)

        #############################
        ## Maintenance
        if self.use_fstar:
            self.state['f_star'] = self.lb
        
        # If you want to track the adaptive step size term, activate the following line.
        # self.state['step_size_list'].append(t1)

        return loss