import numpy as np


class DPAverage(object):
    """
    """
    def __init__(self, window1, dropMinNum1, dropMaxNum1, window2, dropMinNum2, dropMaxNum2):
        self.window1 = window1
        self.dropMinNum1 = dropMinNum1
        self.dropMaxNum1 = dropMaxNum1
        self.window2 = window2
        self.dropMinNum2 = dropMinNum2
        self.dropMaxNum2 = dropMaxNum2
    pass
