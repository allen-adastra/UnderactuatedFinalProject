import pydrake.math as pmath
import numpy as np
import math

from utils import unpack_force_vector, unpack_state_vector

class Dynamics:
    def __init__(self, lf, lr, m, Iz, dt):
        self.lf = lf
        self.lr = lr
        self.m = m
        self.Iz = Iz
        self.dt = dt

    def slip_angles(self, xdot, ydot, psidot, delta):
        # https://arxiv.org/pdf/1703.01225.pdf
        # By convention, positive slip angle shall correspond to positive tire force.
        front = delta - pmath.atan2(ydot + self.lf*psidot, xdot)
        rear = -pmath.atan2(ydot - self.lr*psidot, xdot)
        return front, rear
    
    def simulate(self, state, delta, f_kappa, r_kappa, pacejka_model, dt):
        """

        Args:
            state (1D numpy array): initial state [xdot, ydot, psi, psidot, X, Y]
            delta (float): steer angle
            f_kappa (float): front slip ratio
            r_kappa (float): rear slip ratio
            pacejka_model (function): signature (slip_ratio, slip_angle) -> (Flong, Flat)
        """
        k1, forces = self.f(state, delta, f_kappa, r_kappa, pacejka_model)
        k2, _ = self.f(state + dt * 0.5 * k1, delta, f_kappa, r_kappa, pacejka_model)
        k3, _ = self.f(state + dt * 0.5 * k2, delta, f_kappa, r_kappa, pacejka_model)
        k4, _ = self.f(state + dt * k3, delta, f_kappa, r_kappa, pacejka_model)

        new_state = state + (1.0/6.0)*dt*(k1 + 2.0*k2 + 2.0*k3 + k4)

        return new_state, forces

    def f(self, state, delta, f_kappa, r_kappa, pacejka_model):
        """

        Args:
            state (numpy array)
            psiddot ([type]): IMPORTANT: psiddot is constant at each RK step.
            Fx ([type]): [description]
            Fy ([type]): [description]
        """
        xdot = state[0]
        ydot = state[1]
        psi = state[2]
        psidot = state[3]
        f_alpha, r_alpha = self.slip_angles(xdot, ydot, psidot, delta)
        F_f_long, F_f_lat = pacejka_model(f_kappa, f_alpha)
        F_r_long, F_r_lat = pacejka_model(r_kappa, r_alpha)

        # Trigonometric functions of variables
        cos_d = math.cos(delta)
        sin_d = math.sin(delta)
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        # Net longitudinal and lateral forces.
        Fx = F_r_long + F_f_long*cos_d - F_f_lat*sin_d
        Fy = F_r_lat + F_f_long*sin_d + F_f_lat*cos_d

        derivs = np.array([
                    psidot*ydot + (Fx/self.m),  #xddot
                    -psidot*xdot + (Fy/self.m), #yddot
                    psidot,                     #psidot
                    (1.0/self.Iz)*(self.lf*(F_f_long*sin_d + F_f_lat*cos_d) - self.lr*F_r_lat), #psiddot
                    xdot*cos_psi - ydot*sin_psi, #Xdot
                    xdot*sin_psi + ydot*cos_psi #Ydot
                    ])
        forces = {"f_long" : F_f_long, "f_lat" : F_f_lat, "r_long" : F_r_long, "r_lat" : F_r_lat}
        return derivs, forces

    def nominal_dynamics(self, s0, s1, F, delta):
        """ Return the residuals for the nominal dynamics of the vehicle.

        Args:
            s0 (1D array of ContinuousVariable): [description]
            s1 (1D array of ContinuousVariable): [description]
            F (1D array of ContinuousVariable): [description]
            delta (ContinuousVariable): [description]

        Returns:
            [type]: [description]
        """
        s = unpack_state_vector(s0)
        snext = unpack_state_vector(s1)
        F = unpack_force_vector(F)

        # Trigonometric functions of variables.
        cos_d = pmath.cos(delta)
        sin_d = pmath.sin(delta)
        cos_psi = pmath.cos(s["psi"])
        sin_psi = pmath.sin(s["psi"])

        # Net longitudinal and lateral forces.
        Fx = F["r_long"] + F["f_long"]*cos_d - F["f_lat"]*sin_d
        Fy = F["r_lat"] + F["f_long"]*sin_d + F["f_lat"]*cos_d

        derivs = np.array([
                            s["psidot"]*s["ydot"] + (Fx/self.m),  #xddot
                            -s["psidot"]*s["xdot"] + (Fy/self.m), #yddot
                            s["psidot"],                     #psidot
                            (1.0/self.Iz)*(self.lf*(F["f_long"]*sin_d + F["f_lat"]*cos_d) - self.lr*F["r_lat"]), #psiddot
                            s["xdot"]*cos_psi - s["ydot"]*sin_psi, #Xdot
                            s["xdot"]*sin_psi + s["ydot"]*cos_psi #Ydot
                            ])
        # Apply Euler Integration to get Residuals
        residuals = np.array([
                                snext["xdot"] - (s["xdot"] + self.dt*derivs[0]),
                                snext["ydot"] - (s["ydot"] + self.dt*derivs[1]),
                                snext["psi"] - (s["psi"] + self.dt*derivs[2]),
                                snext["psidot"] - (s["psidot"] + self.dt*derivs[3]),
                                snext["X"] - (s["X"] + self.dt*derivs[4]),
                                snext["Y"] - (s["Y"] + self.dt*derivs[5])
                                ])
        return residuals