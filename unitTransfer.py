import numpy as np


class motorReunit(object):
    """
    arg:
        stream: the data stream filepath
    """
    def __init__(self, stream):
        self.stream = np.loadtxt(stream)
        self.time = 0

    def new(self):
        angle = self.stream[self.time, 0]
        velocity = self.stream[self.time, 2] * 2 * np.pi
        control = (self.stream[self.time, 1] * 2 * np.pi - velocity)  # indicate acceleration by increment
        self.time = self.time + 1
        if self.time == len(self.stream):
            return False
        else:

            return [angle, velocity], control