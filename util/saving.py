import torch

def save_checkpoint(state, is_best=True, is_checkpoint=True, filename='checkpoint.pth.tar'):
    """
    Save the training model if it has the best val performance
    """
    if is_checkpoint:
        torch.save(state, filename)
    else:
        if is_best:
            torch.save(state, filename)