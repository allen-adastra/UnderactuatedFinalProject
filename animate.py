import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D
import matplotlib.collections as clt
import numpy as np
import math
import utils
from visualize import plot_puddles


def generate_animation(xs, ys, psis, steers, physical_params, dt, puddle_model = None):
    # Initialize the figure and artists
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if puddle_model:
        plot_puddles(ax, puddle_model)

    cg_to_fa, = ax.plot([],[], color="k")
    cg_to_ra, = ax.plot([],[], color="k")
    patch_front = Rectangle((0.0, 0.0), width=physical_params.wheel_length, height=physical_params.wheel_width, color="k")
    patch_rear = Rectangle((0.0, 0.0), width=physical_params.wheel_length, height=physical_params.wheel_width, color="k")
    ax.add_patch(patch_front)
    ax.add_patch(patch_rear)

    def init():
        ax.set_xlim(-2, 10)
        ax.set_ylim(0, 10)
        return cg_to_fa, cg_to_ra, patch_front, patch_rear

    def animate(i):
        ax.set_xlim(-2, 10)
        ax.set_ylim(0, 10)

        # Centers of the front wheel x and y
        front_wheel_xy = np.array([xs[i], ys[i]]) + physical_params.lf * np.array([math.cos(psis[i]), math.sin(psis[i])])
        rear_wheel_xy = np.array([xs[i], ys[i]]) - physical_params.lr * np.array([math.cos(psis[i]), math.sin(psis[i])])
        front_wheel_xy_bl = utils.center_to_botleft(front_wheel_xy, psis[i] + steers[i], physical_params.wheel_length, physical_params.wheel_width)
        rear_wheel_xy_bl = utils.center_to_botleft(rear_wheel_xy, psis[i], physical_params.wheel_length, physical_params.wheel_width)

        # Update the patches by using transforms.
        t1 = Affine2D().rotate(psis[i] + steers[i])
        t1.translate(front_wheel_xy_bl[0], front_wheel_xy_bl[1])
        patch_front.set_transform(t1 + ax.transData)

        t2 = Affine2D().rotate(psis[i])
        t2.translate(rear_wheel_xy_bl[0], rear_wheel_xy_bl[1])
        patch_rear.set_transform(t2 + ax.transData)

        # Update the lines.
        cg_to_fa.set_data([xs[i], front_wheel_xy[0]], [ys[i], front_wheel_xy[1]])
        cg_to_ra.set_data([xs[i], rear_wheel_xy[0]], [ys[i], rear_wheel_xy[1]])
        return cg_to_fa, cg_to_ra, patch_front, patch_rear

    ani = FuncAnimation(fig, animate, frames=steers.size, interval=1e3*dt, blit=True)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    writer = FFMpegWriter(fps=1.0/dt)
    # ani.save("animation.mp4", writer=writer, dpi=100)
    plt.show()