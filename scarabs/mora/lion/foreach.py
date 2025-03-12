from __future__ import annotations
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions


def exists(val):
    return val is not None


# class


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert all(
            [
                hasattr(torch, f"_foreach_{attr}_")
                for attr in ("mul", "add", "sign", "lerp")
            ]
        ), "this version of torch does not have the prerequisite foreach functions"

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd, beta1, beta2, decoupled_wd, init_lr = (
                group["lr"],
                group["weight_decay"],
                *group["betas"],
                self.decoupled_wd,
                self._init_lr,
            )

            # maybe decoupled weight decay

            if decoupled_wd:
                wd /= init_lr

            # accumulate List[Tensor] for foreach inplace updates

            params = []
            grads = []
            exp_avgs = []

            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, state = p.grad, self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                params.append(p)
                grads.append(grad)
                exp_avgs.append(exp_avg)

            # stepweight decay

            torch._foreach_mul_(params, 1.0 - lr * wd)

            # weight update

            updates = [t.clone() for t in exp_avgs]
            torch._foreach_lerp_(updates, grads, 1.0 - beta1)
            torch._foreach_sign_(updates)

            torch._foreach_add_(params, updates, alpha=-lr)

            # decay momentum running average

            torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta2)

        return loss
