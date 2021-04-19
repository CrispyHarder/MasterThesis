import warnings
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

def get_lr(optimizer):
    '''assuming there is only one parameter group, 
    returns the lr of the optimizer
    Args:
        optimizer(torch.optim.optimizer): the optimizer
    
    Returns(float): the learning rate of the optimizer'''
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MultistepMultiGammaLR(_LRScheduler):
    """Decays the learning rate of each parameter group by a different gamma,
    (which can be different every time in this adaption of MultiStepLR) once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (list(float)): Multiplicative factor of learning rate decay.
            One for every milestone
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example: with same gamma every time 
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.counter = -1
        super(MultistepMultiGammaLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        
        if not self._get_lr_called_within_step: # pylint: disable=no-member
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
                