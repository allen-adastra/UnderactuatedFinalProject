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