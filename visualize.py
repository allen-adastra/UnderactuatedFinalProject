import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_planned_trajectory(ax, xs, ys, headings, x_length, y_length, fc = 'r'):
    """
    Plot planned trajectory with the ellipse around the vehicle.
    Args:
        xs (arraylike of doubles) : x positions in the global frame
        ys (arraylike of doubles) : y positions in the global frame
        headings (arraylike of doubles) : headings in the global frame
        Q (2 x 2 array) : matrix parameterizing the ellipse in the ego vehicle frame
    """
    ax.plot(xs, ys)
    for x, y, heading in zip(xs, ys, headings):
        ellipse = Ellipse(xy = (x, y), width = x_length, height = y_length, angle = np.rad2deg(heading), alpha = 0.4, ec = "k", fc = fc)
        ax.add_patch(ellipse)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.axis('equal')