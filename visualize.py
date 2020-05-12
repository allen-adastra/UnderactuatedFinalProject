import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
import math
import utils

def plot_planned_trajectory(ax, xs, ys, headings, steers, physical_params, interval = 20):
    """
    Plot planned trajectory with the ellipse around the vehicle.
    Args:
        xs (arraylike of doubles) : x positions in the global frame
        ys (arraylike of doubles) : y positions in the global frame
        headings (arraylike of doubles) : headings in the global frame
        Q (2 x 2 array) : matrix parameterizing the ellipse in the ego vehicle frame
    """
    ax.plot(xs, ys, color="r")
    for i in range(len(steers)):
        # ellipse = Ellipse(xy = (x, y), width = x_length, height = y_length, angle = np.rad2deg(heading), alpha = 0.4, ec = "k", fc = fc)
        # ax.add_patch(ellipse)
        if i % interval == 0:
            plot_vehicle(ax, xs[i], ys[i], headings[i], steers[i], 0.7, 0.7, physical_params.wheel_length, physical_params.wheel_width)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.axis('equal')

def plot_slips(all_slips):
    """[summary]

    Args:
        all_slips (list of instances of SlipParameters): [description]
    """
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    kappa_fs = [s.kappa_f for s in all_slips]
    kappa_rs = [s.kappa_r for s in all_slips]
    alpha_fs = [s.alpha_f for s in all_slips]
    alpha_rs = [s.alpha_r for s in all_slips]
    ax1.plot(kappa_fs)
    ax1.plot(kappa_rs)
    ax1.set_title("Slip Ratios")
    ax1.legend(["Front", "Rear"])

    ax2.plot(alpha_fs)
    ax2.plot(alpha_rs)
    ax2.set_title("Slip Angles")
    ax2.legend(["Front", "Rear"])
    plt.show()

def plot_forces(forces):
    """[summary]

    Args:
        forces (T by 4 array): [description]
    """
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    F_f_long = np.zeros(forces.shape[0])
    F_f_lat = np.zeros(forces.shape[0])
    F_r_long = np.zeros(forces.shape[0])
    F_r_lat = np.zeros(forces.shape[0])
    for i in range(forces.shape[0]):
        f = utils.unpack_force_vector(forces[i])
        F_f_long[i] = f["f_long"]
        F_f_lat[i] = f["f_lat"]
        F_r_long[i] = f["r_long"]
        F_r_lat[i] = f["r_lat"]
    ax1.plot(F_f_long)
    ax1.plot(F_r_long)
    ax1.set_title("Longitudinal forces.")
    ax1.legend(["Front", "Rear"])

    ax2.plot(F_f_lat)
    ax2.plot(F_r_lat)
    ax2.set_title("Lateral forces.")
    ax2.legend(["Front", "Rear"])
    plt.show()

def plot_vehicle(ax, cg_x, cg_y, psi, delta, lf, lr, wheel_length, wheel_width):
    # Centers of the front wheel x and y
    front_wheel_xy = np.array([cg_x, cg_y]) + lf * np.array([math.cos(psi), math.sin(psi)])
    rear_wheel_xy = np.array([cg_x, cg_y]) - lr * np.array([math.cos(psi), math.sin(psi)])
    front_wheel_xy_bl = utils.center_to_botleft(front_wheel_xy, psi + delta, wheel_length, wheel_width)
    rear_wheel_xy_bl = utils.center_to_botleft(rear_wheel_xy, psi, wheel_length, wheel_width)

    # A bit counterintuitive, but width of the rectangle corresponds to wheel length and height
    # corresponds to wheel width.
    front_wheel = Rectangle(front_wheel_xy_bl, width = wheel_length, height = wheel_width, angle = np.rad2deg(psi + delta), color="k")
    rear_wheel = Rectangle(rear_wheel_xy_bl, width = wheel_length, height = wheel_width, angle = np.rad2deg(psi), color="k")

    ax.add_patch(front_wheel)
    ax.add_patch(rear_wheel)
    ax.plot([cg_x, front_wheel_xy[0]], [cg_y, front_wheel_xy[1]], color="k")
    ax.plot([cg_x, rear_wheel_xy[0]], [cg_y, rear_wheel_xy[1]], color="k")

def plot_puddles(ax, puddle_model):
    for center, shape in zip(puddle_model.centers, puddle_model.shapes):
        eigvals, eigvecs = np.linalg.eig(shape)
        x_length = eigvals[0]
        y_length = eigvals[1]
        angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1]) * 180 / np.pi
        ell = Ellipse(center, x_length, y_length, -angle)
        ax.add_patch(ell)

def plot_ellipse_arc(ax, ea, n_pts = 50):
    ts = np.linspace(ea.start_t, ea.end_t, n_pts)
    out = [ea.eval_parametric(t) for t in ts]
    xs = [o[0] for o in out]
    ys = [o[1] for o in out]
    ax.plot(xs, ys, "k--", linewidth=2)