import numpy as np


class DPAverage(object):
    """
    args:
        windowLength1: the length of outer loop window
        dropMinNum1: mins to drop in the outer loop window
        dropMaxNum1: maxes to drop in the outer loop window
        windowLength2: the length of inner loop window
        dropMinNum2: mins to drop in the inner loop window
        dropMaxNum2: maxes to drop in inner loop window
    """
    def __init__(self, windowLength1, dropMinNum1, dropMaxNum1, windowLength2, dropMinNum2, dropMaxNum2):
        self.window = []  # length of windowLength1
        self.intermedate = []  # length of windowLength2
        self.windowLength1 = windowLength1
        self.dropMinNum1 = dropMinNum1
        self.dropMaxNum1 = dropMaxNum1
        self.windowLength2 = windowLength2
        self.dropMinNum2 = dropMinNum2
        self.dropMaxNum2 = dropMaxNum2

    def average(self, window, windowLength, dropMinNum, dropMaxNum):
        """
        """
        if len(window) == (windowLength + 1):
            window.pop(0)
            windowSort = window.copy()
            windowSort.sort()
            windowSort = windowSort[dropMinNum:(len(window)-dropMaxNum)]
            sum = 0
            for item in windowSort:
                sum = sum + item
            return sum / (len(window) - dropMinNum - dropMaxNum)
        else:
            return window[-1]

    def new(self, data):
        """
        """
        # update data in outer window
        self.window.append(data)
        # update intermedate stream
        self.intermedate.append(self.average(self.window, self.windowLength1, self.dropMinNum1, self.dropMaxNum1))
        return self.average(self.intermedate, self.windowLength2, self.dropMinNum2, self.dropMaxNum2)
