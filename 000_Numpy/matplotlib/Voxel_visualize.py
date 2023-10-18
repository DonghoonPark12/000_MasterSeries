"""
==========================
3D voxel / volumetric plot
==========================

Demonstrates plotting 3D volumetric objects with `.Axes3D.voxels`.
"""

# import matplotlib.pyplot as plt
# import numpy as np

# # prepare some coordinates
# x, y, z = np.indices((8, 8, 8))

# # draw cuboids in the top left and bottom right corners, and a link between
# # them
# cube1 = (x < 3) & (y < 3) & (z < 3)
# cube2 = (x >= 5) & (y >= 5) & (z >= 5)
# link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# # combine the objects into a single boolean array
# voxelarray = cube1 | cube2 | link

# # set the colors of each object
# colors = np.empty(voxelarray.shape, dtype=object)
# colors[link] = 'red'
# colors[cube1] = 'blue'
# colors[cube2] = 'green'

# # and plot everything
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

# plt.show()


# def midpoints(x):
#     sl = ()
#     for _ in range(x.ndim):
#         x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
#         sl += np.index_exp[:]
#     return x

# # prepare some coordinates, and attach rgb values to each
# r, g, b = np.indices((17, 17, 17)) / 16.0
# rc = midpoints(r)
# gc = midpoints(g)
# bc = midpoints(b)

# # define a sphere about [0.5, 0.5, 0.5]
# sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

# # combine the color components
# colors = np.zeros(sphere.shape + (3,))
# colors[..., 0] = rc
# colors[..., 1] = gc
# colors[..., 2] = bc

# # and plot everything
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(r, g, b, sphere,
#           facecolors=colors,
#           edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
#           linewidth=0.5)
# ax.set(xlabel='r', ylabel='g', zlabel='b')
# ax.set_aspect('equal')

# plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_voxel(ax, position, color='b', size=1):
    # Plot a voxel at the specified position with the specified color and size
    ax.bar3d(position[0], position[1], position[2], dx=size, dy=size, dz=size, color=color)

# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Example: Plotting three voxels at different positions and colors
plot_voxel(ax, (1, 1, 1), color='b')
plot_voxel(ax, (2, 2, 2), color='g')
plot_voxel(ax, (3, 3, 3), color='r')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()