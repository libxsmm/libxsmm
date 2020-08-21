import torch
from torch.optim.optimizer import Optimizer, required

_available = False
try:
    from pcl_embedding_bag import bf16_update
    _available = True
except ImportError as e:
    #print(e)
    pass

def is_available():
    return _available

class SplitSGD(Optimizer):
    r"""Implements low precision stochastic gradient descent with extra state."""

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not is_available():
            raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum != 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay != 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SplitSGD, self).__init__(params, defaults)
        print("Using SplitSGD")

    def __setstate__(self, state):
        super(SplitSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dtype == torch.bfloat16:
                    param_state = self.state[p]
                    if 'low_bits' not in param_state:
                        buf = param_state['low_bits'] = torch.zeros_like(p.data, dtype=torch.short)
                    else:
                        buf = param_state['low_bits']
                # if weight_decay != 0:
                #     d_p = d_p.add(weight_decay, p.data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                #p.data.add_(-group['lr'], d_p)
                if p.dtype == torch.bfloat16:
                    bf16_update(p.data, buf, d_p, -group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])

        return loss
