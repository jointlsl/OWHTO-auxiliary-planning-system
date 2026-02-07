from .optim import *


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,

    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,

    }[s.lower()]
