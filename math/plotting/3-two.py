#!/usr/bin/env python3
"""
Script for plotting the exponential decay of radioactive elements
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, 'r--', label='C-14')  # plot y1 against x as a dashed red line
plt.plot(x, y2, 'g-', label='Ra-226') # plot y2 against x as a solid green line

# labels, title and legend
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.legend(loc='upper right')

# setting the limits for x and y axes
plt.xlim(0, 20000)
plt.ylim(0, 1)

plt.show()  # display the plot
