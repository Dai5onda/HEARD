import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV files
accel_L_data = pd.read_csv('accel_L.csv')
accel_R_data = pd.read_csv('accel_R.csv')

# Replace Inf values with NaNs and drop rows with NaN values for both datasets
accel_L_data.replace([np.inf, -np.inf], np.nan, inplace=True)
accel_R_data.replace([np.inf, -np.inf], np.nan, inplace=True)
accel_L_data.dropna(inplace=True)
accel_R_data.dropna(inplace=True)

# Extract time and XYZ coordinates
time_L = accel_L_data['Time'].values
xyz_L = accel_L_data[['x', 'y', 'z']].values
time_R = accel_R_data['Time'].values
xyz_R = accel_R_data[['x', 'y', 'z']].values
downsample_rate = 2  # Only take every 2nd data point
xyz_L_downsampled = xyz_L[::downsample_rate]
xyz_R_downsampled = xyz_R[::downsample_rate]

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set up the axis limits
ax.set_xlim(min(np.concatenate([xyz_L[:,0], xyz_R[:,0]])), max(np.concatenate([xyz_L[:,0], xyz_R[:,0]])))
ax.set_ylim(min(np.concatenate([xyz_L[:,1], xyz_R[:,1]])), max(np.concatenate([xyz_L[:,1], xyz_R[:,1]])))
ax.set_zlim(min(np.concatenate([xyz_L[:,2], xyz_R[:,2]])), max(np.concatenate([xyz_L[:,2], xyz_R[:,2]])))

# Set the labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot of Accelerometer Data')

# Initialize two scatter plots in the same figure for left and right sensors
scatter_L = ax.scatter([], [], [], c='blue', marker='o', label='Left Sensor')
scatter_R = ax.scatter([], [], [], c='red', marker='^', label='Right Sensor')
point_counter = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# Animation update function
def update(num):
    scatter_L._offsets3d = (xyz_L[:num, 0], xyz_L[:num, 1], xyz_L[:num, 2])
    scatter_R._offsets3d = (xyz_R[:num, 0], xyz_R[:num, 1], xyz_R[:num, 2])
    point_counter.set_text(f'Point Index: {num}')
    return scatter_L, scatter_R, point_counter

# Create the animation
ani = FuncAnimation(fig, update, frames=min(len(time_L), len(time_R)), interval=0.0004, repeat=True)

# Show the legend
ax.legend()

# Show the animation
plt.show()
