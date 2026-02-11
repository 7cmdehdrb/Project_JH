import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_spline_course(x, y, ds=0.1):
    """
    calc_spline_course function from the notebook.
    Requires Spline2D class to be defined.
    """
    import bisect
    import math

    class Spline:
        """Cubic Spline class"""

        def __init__(self, x, y):
            self.b, self.c, self.d, self.w = [], [], [], []
            self.x = x
            self.y = y
            self.nx = len(x)
            h = np.diff(x)
            self.a = [iy for iy in y]
            A = self.__calc_A(h)
            B = self.__calc_B(h)
            self.c = np.linalg.solve(A, B)
            for i in range(self.nx - 1):
                self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
                tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (
                    self.c[i + 1] + 2.0 * self.c[i]
                ) / 3.0
                self.b.append(tb)

        def calc(self, t):
            if t < self.x[0] or t > self.x[-1]:
                return None
            i = self.__search_index(t)
            dx = t - self.x[i]
            result = (
                self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0
            )
            return result

        def calcd(self, t):
            if t < self.x[0] or t > self.x[-1]:
                return None
            i = self.__search_index(t)
            dx = t - self.x[i]
            result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0
            return result

        def calcdd(self, t):
            if t < self.x[0] or t > self.x[-1]:
                return None
            i = self.__search_index(t)
            dx = t - self.x[i]
            result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
            return result

        def __search_index(self, x):
            return bisect.bisect(self.x, x) - 1

        def __calc_A(self, h):
            A = np.zeros((self.nx, self.nx))
            A[0, 0] = 1.0
            for i in range(self.nx - 1):
                if i != (self.nx - 2):
                    A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
                A[i + 1, i] = h[i]
                A[i, i + 1] = h[i]
            A[0, 1] = 0.0
            A[self.nx - 1, self.nx - 2] = 0.0
            A[self.nx - 1, self.nx - 1] = 1.0
            return A

        def __calc_B(self, h):
            B = np.zeros(self.nx)
            for i in range(self.nx - 2):
                B[i + 1] = (
                    3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1]
                    - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
                )
            return B

    class Spline2D:
        """2D Cubic Spline class"""

        def __init__(self, x, y):
            self.s = self.__calc_s(x, y)
            self.sx = Spline(self.s, x)
            self.sy = Spline(self.s, y)

        def __calc_s(self, x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            self.ds = np.hypot(dx, dy)
            s = [0]
            s.extend(np.cumsum(self.ds))
            return s

        def calc_position(self, s):
            x = self.sx.calc(s)
            y = self.sy.calc(s)
            return x, y

        def calc_curvature(self, s):
            dx = self.sx.calcd(s)
            ddx = self.sx.calcdd(s)
            dy = self.sy.calcd(s)
            ddy = self.sy.calcdd(s)
            k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
            return k

        def calc_yaw(self, s):
            dx = self.sx.calcd(s)
            dy = self.sy.calcd(s)
            yaw = math.atan2(dy, dx)
            return yaw

    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def plot_error(player_df: pd.DataFrame, waypoint_df: pd.DataFrame):
    """
    Plot error metrics between player trajectory and waypoint spline path

    Parameters:
    -----------
    player_df : pd.DataFrame
        DataFrame containing player position data with columns:
        - 'time': timestamp
        - 'pose.position.x', 'pose.position.y', 'pose.position.z': position coordinates

    waypoint_df : pd.DataFrame
        DataFrame containing waypoint data with columns:
        - 'poses[i].position.x', 'poses[i].position.y', 'poses[i].position.z'
          for i in range(12)
    """
    # Extract player positions
    times = player_df["time"].to_numpy()
    player_xs = player_df["pose.position.x"].to_numpy()
    player_ys = player_df["pose.position.y"].to_numpy()
    player_zs = player_df["pose.position.z"].to_numpy()

    # Extract waypoint positions (12 waypoints)
    waypoint_xs = [
        waypoint_df[f"poses[{i}].position.x"].to_numpy()[0] for i in range(12)
    ]
    waypoint_ys = [
        waypoint_df[f"poses[{i}].position.y"].to_numpy()[0] for i in range(12)
    ]
    waypoint_zs = [
        waypoint_df[f"poses[{i}].position.z"].to_numpy()[0] for i in range(12)
    ]

    # Create spline path from waypoints (2D for XY plane)
    rx, ry, ryaw, rk, s = calc_spline_course(waypoint_xs, waypoint_ys, ds=0.01)
    spline_points = np.column_stack([rx, ry])

    # Calculate minimum distance from each player position to spline path
    distances = []
    for px, py in zip(player_xs, player_ys):
        player_point = np.array([px, py])
        dists_to_spline = np.linalg.norm(spline_points - player_point, axis=1)
        min_dist = np.min(dists_to_spline)
        distances.append(min_dist)

    distances = np.array(distances)

    # Calculate L2 norm of entire trajectory error
    l2_norm = np.sqrt(np.sum(distances**2))
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    std_error = np.std(distances)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Distance over time
    ax1.plot(times, distances, color="blue", linewidth=2, label="Distance to path")
    ax1.axhline(
        y=mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.3f}m"
    )
    ax1.fill_between(times, 0, distances, alpha=0.3)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance to Spline Path (m)")
    ax1.set_title("Player Distance from Reference Path Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add text with metrics
    textstr = f"L2 Norm: {l2_norm:.2f}\nMean Error: {mean_error:.3f}m\nMax Error: {max_error:.3f}m"
    ax1.text(
        0.02,
        0.98,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 2: 2D trajectory comparison
    ax2.plot(
        waypoint_xs,
        waypoint_ys,
        "ro-",
        markersize=10,
        linewidth=2,
        label="Waypoints",
        alpha=0.7,
    )
    ax2.plot(rx, ry, "g--", linewidth=2, label="Spline Path", alpha=0.7)
    ax2.plot(
        player_xs, player_ys, "b-", linewidth=1.5, label="Player Trajectory", alpha=0.8
    )

    # Mark start and end
    ax2.plot(player_xs[0], player_ys[0], "go", markersize=12, label="Start")
    ax2.plot(player_xs[-1], player_ys[-1], "rs", markersize=12, label="End")

    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_title("Trajectory Comparison (2D)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print error metrics
    print(f"Error Metrics:")
    print(f"  L2 Norm: {l2_norm:.4f}")
    print(f"  Mean Distance: {mean_error:.4f}m")
    print(f"  Max Distance: {max_error:.4f}m")
    print(f"  Std Distance: {std_error:.4f}m")


# Example usage:
# player_df = pd.read_csv("player_pose.csv")
# waypoint_df = pd.read_csv("waypoint_array.csv")
# plot_error(player_df, waypoint_df)
