#!/usr/bin/env python3
"""
Script for plotting a histogram of student scores for a project
"""


import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# plot a histogram
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

# labels and title
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

# set the limits for x and y axes
plt.xlim(0, 100)
plt.ylim(0, 30)

plt.show()  # display the plot
