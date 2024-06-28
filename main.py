import numpy as np
import random
import math
import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib.patches import Circle





num_centers = 5
num_points = 60
map_size = 40
t = 30
n = 5
max_distance = 20
speed = 60
time_limit = 24 * 60 // t
priority_weights = {'一般': 1, '较紧急': 2, '紧急': 3}
population_size = 1000
generations = 40
mutation_rate = 0.3
crossover_rate = 0.9