import matplotlib.pyplot as plt
import numpy as np
from torch import tensor

a = [
    tensor([1.0043e-01, 3.5033e-01, 5.4907e-01])]

alen = len(a)

x = np.arange(0.5, alen, 1)
probabilities = np.vstack(a).T

# plot
fig, ax = plt.subplots()
bottom = np.zeros(len(a))

width = 1

p = ax.bar(x, probabilities[1], width, bottom=bottom, color='limegreen')
bottom += probabilities[1]
p = ax.bar(x, probabilities[2], width, bottom=bottom, color='skyblue')
bottom += probabilities[2]
p = ax.bar(x, probabilities[0], width, bottom=bottom, color='white')

ax.set(xlim=(0, alen), ylim=(0, 1))

plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
ax.axis('off')
plt.savefig(fname='graph.svg', bbox_inches='tight', pad_inches=0, transparent=False)
