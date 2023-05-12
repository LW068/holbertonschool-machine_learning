#!/usr/bin/env python3
"""
This module plots a line graph.
"""


import numpy as np
import matplotlib.pyplot as plt


y = np.arange(0, 11) ** 3


def plot_graph():
    """
    Function to plot a line graph with y as a solid red line
    and the x-axis ranging from 0 to 10.
    """
    plt.plot(y, 'r')  # 'r' specifies a red line
    plt.xlim(0, 10)   # x-axis ranges from 0 to 10
    plt.show()        # display the plot


plot_graph()  # call the function
