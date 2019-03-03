import numpy as np


class reunit(object):
    """
    arg:
        stream: the data stream filepath
    """
    def __init__(self, stream):
        self.stream = np.loadtxt(stream)
        self.time = 0

    def new(self):
        angle = angleIn
        velocity = velocityIn * 2 * np.pi
        control = (controlIn * 2 * np.pi - velocity)  # indicate acceleration by increment

