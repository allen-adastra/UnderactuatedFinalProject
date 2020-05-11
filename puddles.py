import math
import numpy as np
import matplotlib.pyplot as plt
from visualize import plot_puddles

def multi_exponential(xy, center, shape, scale):
  """
  Args:
    center (vector) : center of the multvariate exponential
    shape (matrix) : shape matrix
  """
  return scale * np.exp(-(xy - center).T @ np.linalg.inv(shape) @ (xy - center))

def mean_fraction_fun(center, shape, scale):
  """ Returns a function (x, y) -> mean scaling

  Args:
      center ([type]): [description]
      shape ([type]): [description]
      scale ([type]): [description]

  Returns:
      [type]: [description]
  """
  mean_fraction = lambda x, y: 1.0 - multi_exponential(np.array([x, y]), center, shape, scale)
  return np.vectorize(mean_fraction)

def var_fraction_fun(center, shape, scale):
    """ Returns a function (x, y) -> variance scaling.

    Args:
        center ([type]): [description]
        shape ([type]): [description]
        scale ([type]): [description]

    Returns:
        [type]: [description]
    """
    variance_fraction = lambda x, y: 1.0 + multi_exponential(np.array([x, y]), center, shape, scale)
    return np.vectorize(variance_fraction)

class PuddleModel:
    def __init__(self, centers, shapes, mean_scales, variance_scales):
        """
        Args:
            centers ([type]): [description]
            shapes ([type]): [description]
            mean_scales ([type]): for v in this list, the ultimate scaling is 1.0 - v
            variance_scales ([type]): for v in this list, the ultimate scaling is 1.0 + v
        """
        self.centers = centers
        self.shapes = shapes
        self.mean_scales = [len(self.centers) * s for s in mean_scales] # need to adjust for the number of components
        self.variance_scales = [len(self.centers) * s for s in variance_scales]
        self.mean_component_funcs = [mean_fraction_fun(center, shape, scale) for center, shape, scale in zip(self.centers, self.shapes, self.mean_scales)]
        self.var_component_funcs = [var_fraction_fun(center, shape, scale) for center, shape, scale in zip(self.centers, self.shapes, self.variance_scales)]
        self.mean_fun = lambda x, y : sum([(1.0/len(self.centers)) * fun(x, y) for fun in self.mean_component_funcs])
        self.var_fun = lambda x, y: sum([(1.0/len(self.centers)) * fun(x, y) for fun in self.var_component_funcs])

    def get_mean_fun(self):
        """
        Returns:
            function: (x, y) -> mean scaling
        """
        return self.mean_fun
    
    def get_var_fun(self):
        """
        Returns:
            function: (x, y) -> variance scaling
        """
        return self.var_fun