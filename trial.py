from collections import deque, namedtuple, defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random


def a():
    plt.axis()
    plt.ion()
    for i in range(100):
        y = np.random.random()
        plt.autoscale()
        plt.plot(i, i, '.r')
        plt.plot(i, y, '.b')
        plt.pause(0.01)
    plt.cla()


a()
a()