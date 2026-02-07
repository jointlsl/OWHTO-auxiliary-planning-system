from .loss import *


def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'focal': FocalLossV1,
        'dice': DiceLoss
    }[s.lower()]