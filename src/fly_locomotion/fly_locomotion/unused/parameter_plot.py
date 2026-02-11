import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(ROOT_DIR, "parameter_test/CONTROL2")
file_names = os.listdir(file_path)

linear_velocity_gains = []
angular_velocity_gains = []
passed_waypoints = []
times = []

for file_name in file_names:
    if file_name.endswith(".csv"):
        df = pd.read_csv(os.path.join(file_path, file_name))
        linear_velocity_gains.extend(df['linear_velocity_gain'].tolist())
        angular_velocity_gains.extend(df['angular_velocity_gain'].tolist())
        passed_waypoints.extend(df['passed_waypoint'].tolist())
        times.extend((df['time']).tolist())

# Plot x:linear_velocity_gain, y:angular_velocity_gain, z:passed_waypoint in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(linear_velocity_gains, angular_velocity_gains, times, c='b', cmap='viridis')
ax.set_xlabel('Linear Velocity Gain')
ax.set_ylabel('Angular Velocity Gain')
ax.set_zlabel('Passed Waypoint')
fig.colorbar(scat, ax=ax, label='Passed Waypoint')
plt.title('Parameter Effects on Passed Waypoints')
plt.show()  