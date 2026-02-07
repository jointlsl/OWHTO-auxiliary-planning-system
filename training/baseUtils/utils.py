#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
    :Purpose: Utils Function
    :Authors: Hongkaiz
    :Date   : 2021-12-10
    :Version: 0.1
    :Usage  :
    :Note   :
"""

import time

from loguru  import logger

##########################################
def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    def function_timer(*args, **kwargs):
        logger.info('Start Run Function {name} ...'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        logger.info('Finish Run Function {name} ...'.format(name=function.__name__))
        logger.info('The Function {name} Run Time : {time:.2f}s'.format(name=function.__name__, time = t1 - t0))
        return result
    return function_timer
##ondef