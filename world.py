import math
import numpy as np
import matplotlib.pyplot as plt

class EllipseArc:
    def __init__(self, a, b, x_center, y_center, start_t, end_t):
        self.a = a
        self.b = b
        self.x_center = x_center
        self.y_center = y_center
        self.start_t = start_t
        self.end_t = end_t

    @property
    def semi_major(self):
        return max(self.a, self.b)

    @property
    def semi_minor(self):
        return min(self.a, self.b)

    def approx_dist_fun(self):
        """ Returns a function with the signature (x, y) which returns some notion of distance
        away from the ellipse boundary.
        """
        def fun(x, y):
            xterm = (1.0/self.a**2.0) * (x - self.x_center)**2.0
            yterm = (1.0/self.b**2.0) * (y - self.y_center)**2.0
            return (xterm + yterm - 1)**2.0
        return fun

    def eval_parametric(self, t):
        return self.x_center + self.a * math.cos(t), self.y_center + self.b * math.sin(t)

class Ellipse:
    def __init__(self, a, b, x_center, y_center):
        self.a = a
        self.b = b
        self.x_center = x_center
        self.y_center = y_center
    
    @property
    def semi_major(self):
        return max(self.a, self.b)

    @property
    def semi_minor(self):
        return min(self.a, self.b)

    def approx_dist_fun(self):
        """ Returns a function with the signature (x, y) which returns some notion of distance
        away from the ellipse boundary.
        """
        def fun(x, y):
            xterm = (1.0/self.a**2.0) * (x - self.x_center)**2.0
            yterm = (1.0/self.b**2.0) * (y - self.y_center)**2.0
            return (xterm + yterm - 1)**2.0
        return fun

if __name__ == "__main__":
    ea = EllipseArc(6.0, 3.0, 0, 3.0, -0.5 * math.pi, 0.5 * math.pi)

    ts = np.linspace(-0.5*math.pi, 0.5*math.pi, 100)

    out = [ea.eval_parametric(t) for t in ts]
    xs = [o[0] for o in out]
    ys = [o[1] for o in out]
    plt.plot(xs, ys)
    plt.show()