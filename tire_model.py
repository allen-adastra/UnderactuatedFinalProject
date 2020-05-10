import numpy as np
import math
import matplotlib.pyplot as plt
from utils import array_norm

def approx_arctan(x):
  """ Within the regime of small x, arctan(x) is approximately x.
  Args:
      x ([type]): [description]

  Returns:
      [type]: [description]
  """
  return x

class TireModel:
  def __init__(self, params):
    self._params = params

  def get_combined_slip_model(self, math_pkg):
    """
    Args:
      math_pkg : either math or pydrake.math
    """
    # https://projects.skill-lync.com/projects/Combined-Slip-Brush-Tire-Model-15854

    # Longitudinal parameters.
    Bx = self._params["Bx"]
    Cx = self._params["Cx"]
    Dx = self._params["Dx"]
    Ex = self._params["Ex"]
  
    # Lateral parameters.
    By = self._params["By"]
    Cy = self._params["Cy"]
    Dy = self._params["Dy"]
    Ey = self._params["Ey"]
    gamma = self._params["k_alpha_ratio"]

    def pacejka_model(slip_ratio, slip_angle):
      # Longitudinal forces
      # Modified longitudinal slip ratios
      k_mod = np.linalg.norm([slip_ratio, gamma * slip_angle])
      k_adjusted = slip_ratio/k_mod
      Fx = k_adjusted*Dx*math_pkg.sin(Cx*math_pkg.atan(Bx*k_mod-Ex*(Bx*k_mod-math_pkg.atan(Bx*k_mod))))
      
      # Lateral forces
      # Modified lateral slip angles
      alpha_mod = np.linalg.norm([slip_angle, slip_ratio/gamma])
      alpha_adjusted = slip_angle/alpha_mod
      Fy = alpha_adjusted*Dy*math_pkg.sin(Cy*math_pkg.atan(By*alpha_mod-Ey*(By*alpha_mod-math_pkg.atan(By*alpha_mod))))
      return Fx, Fy

    return pacejka_model

  def plot_cross_sections(self):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    combined_slip_model = self.get_combined_slip_model(math)
    for k in np.linspace(0, 0.08, 4):
      alphas = np.linspace(-0.1, 0.1, 100)

      forces = [combined_slip_model(k, a) for a in alphas]
      fx = [f[0] for f in forces]
      fy = [f[1] for f in forces]
      net_force = [np.linalg.norm(np.asarray(f)) for f in forces]

      ax1.plot(alphas, fx)
      ax2.plot(alphas, fy)
      ax3.plot(alphas, net_force)

  def plot_surface(self):
      """
      Plot the total force acting on the car as a function of slip angle and ratio.
      """
      slip_angles = np.arange(-0.15, 0.15, 0.005)
      slip_ratios = np.arange(-0.15, 0.15, 0.005)
      slip_ratios, slip_angles = np.meshgrid(slip_ratios, slip_angles)
      combined_slip_model = self.get_combined_slip_model(math)
      combined_slip_model = np.vectorize(combined_slip_model)
      force_vectors = combined_slip_model(slip_ratios, slip_angles)

      net_force = array_norm(force_vectors[0], force_vectors[1])

      fig = plt.figure()
      ax = fig.gca(projection='3d')
      ax.set_xlabel("Slip Ratio")
      ax.set_ylabel("Slip Angle (rad)")
      ax.set_zlabel("Force (N)")
      ax.set_title("Net Force Acting on the Tire")
      plt.set_cmap("viridis_r")
      res = ax.contour3D(slip_ratios, slip_angles, net_force, 50)


if __name__ == "__main__":
    params = {"Bx":25, "Cx":2.1, "Dx":3200, "Ex":-0.4, 
              "By":15.5, "Cy":2.0, "Dy":2900, "Ey":-1.6, "k_alpha_ratio":9.0/7.0}
    m = TireModel(params)
    m.plot_cross_sections()
    plt.show()