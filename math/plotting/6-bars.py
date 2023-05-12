#!/usr/bin/env python3
"""
Script for plotting a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define the color for each type of fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Define the labels for each type of fruit
labels = ['apples', 'bananas', 'oranges', 'peaches']

# Define the labels for each person
persons = ['Farrah', 'Fred', 'Felicia']

# Define the bar width
bar_width = 0.5

# Define the cumulative sum along the rows
y_offset = np.zeros(fruit.shape[1])

# Create the bars
for row in range(fruit.shape[0]):
    plt.bar(persons, fruit[row], bar_width, bottom=y_offset,
            color=colors[row], label=labels[row])
    y_offset += fruit[row]

plt.xlabel('Persons')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
