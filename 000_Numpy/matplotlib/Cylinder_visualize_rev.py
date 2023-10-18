import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

cylinder_radius = 20
cylinder_height = 25
pcd_position = (0, 0, -2)
pcd_size = (30, 30, 20)

# Cylinder
def plot_cylinder(ax, radius=1, height=30, resolution=100):
    min_height, max_height = -5, height-5
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(min_height, max_height, resolution)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=0.5, color='b')

    # Cylinder의 아랫면
    ax.plot_surface(5 * np.cos(theta), 5 * np.sin(theta), np.full_like(z, min_height), alpha=0.5, color='g')
    # Cylinder의 윗면
    ax.plot_surface(5 * np.cos(theta), 5 * np.sin(theta), np.full_like(z, max_height), alpha=0.5, color='g')

# Cube
# def plot_cube(ax, center=(0, 0, 0), size=1, color='b'):
#     axes = [30, 30, 20] # Create axis
#     data = np.ones(axes, dtype=bool) # Create Data
#     colors = np.empty(axes + [4], dtype=np.float32) # Control colour
    
#     colors[:] = [1, 0, 0, 0.9]  # red + Transparency
#     ax.voxels(data, facecolors=colors)

# 3D Bar
def plot_bar(ax, position, color='b', size=(0,0,0)):
    # Plot a voxel at the specified position with the specified color and size
    ax.bar3d(position[0]-size[0]/2, position[1]-size[1]/2, position[2],
             dx=size[0], dy=size[1], dz=size[2], color=color, alpha=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_cylinder(ax, radius=cylinder_radius, height=cylinder_height)
#plot_cube(ax, center=(0, 0, 5), size=2, color='b')
plot_bar(ax, pcd_position, color='r', size=pcd_size)

# Set axis limits
ax.set_xlim([-40, 40])
ax.set_ylim([-40, 40])
ax.set_zlim([-10, 35])

# Show plot
plt.show()
