import warnings
from collections import Counter
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


class MultistepMultiGammaLR(_LRScheduler):
    
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.counter = -1
        super(MultistepMultiGammaLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        #if we change lr, increase counter by one
        self.counter += 1
        return [group['lr'] * self.gamma[self.counter]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        lr_modifier_total = 1
        for i in range(self.counter):
            lr_modifier_total = lr_modifier_total * self.gamma[i]
        return [base_lr * self.gamma ** lr_modifier_total
                for base_lr in self.base_lrs]