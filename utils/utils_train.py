import torch

def correct_cal(logit, target):
    pred = torch.argmax(logit, dim=1)
    judge = (pred==target)
    count = torch.sum(judge).item()
    return count