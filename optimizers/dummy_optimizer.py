import itertools
import torch


class DummyMinimaxOptimizer(torch.optim.Optimizer):
    """
    Optimizer which does nothing but has minimax API.
    """
    def __init__(self, f_params, g_params, f_lr: float, g_lr: float):
        assert f_lr > 0 and g_lr > 0
        self.f_params = f_params
        self.g_params = g_params
        params = list(itertools.chain(f_params, g_params))
        defaults = dict(f_lr=f_lr, g_lr=g_lr)

        super(DummyMinimaxOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        return loss
