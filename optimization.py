import numpy as np
import math
from pydrake.all import MathematicalProgram, SnoptSolver, IpoptSolver, SolverOptions
import pydrake.math as pmath
import matplotlib.pyplot as plt

from dynamics import Dynamics
from tire_model import TireModel
from utils import unpack_force_vector, unpack_state_vector, pack_state_vector, pack_force_vector, extract_time_series
from world import Ellipse
from visualize import plot_planned_trajectory

class Optimization:
    def __init__(self, config):
        self.lf = config["lf"]
        self.lr = config["lr"]
        self.m = config["m"]
        self.Iz = config["Iz"]
        self.T = config["T"]
        self.dt = config["dt"]
        self.initial_state = config["xinit"]
        self.goal_state = config["xgoal"]
        self.ellipse = config["ellipse"]
        self.deviation_cost = config["deviation_cost"]
        self.Qf = config["Qf"]
        self.min_xdot = config["min_xdot"]
        self.min_slip_ratio_mag = config["min_slip_ratio_mag"]
        self.max_ddelta = config["max_ddelta"]
        self.max_dkappa = config["max_dkappa"]
        self.max_delta = config["max_delta"]
        self.n_state = 6
        self.n_nominal_forces = 4
        self.pacejka_params = config["pacejka_params"]
        self.dynamics = Dynamics(self.lf, self.lr, self.m, self.Iz, self.dt)
        self.tire_model = TireModel(self.pacejka_params)
    
    def build_program(self):
        self.prog = MathematicalProgram()

        # Declare variables.
        state = self.prog.NewContinuousVariables(rows=self.T+1, cols=self.n_state, name='state')
        nominal_forces = self.prog.NewContinuousVariables(rows=self.T, cols=self.n_nominal_forces, name='nominal_forces')
        steers = self.prog.NewContinuousVariables(rows=self.T, name="steers")
        slip_ratios = self.prog.NewContinuousVariables(rows=self.T, cols=2, name="slip_ratios")
        self.state = state
        self.nominal_forces = nominal_forces
        self.steers = steers
        self.slip_ratios = slip_ratios

        # Set the initial state.
        xinit = pack_state_vector(self.initial_state)
        for i, s in enumerate(xinit):
            self.prog.AddConstraint(state[0, i] == s)

        # Constrain xdot to always be at least some value to prevent numerical issues with optimizer.
        for i in range(self.T + 1):
            s = unpack_state_vector(state[i])
            self.prog.AddConstraint(s["xdot"] >= self.min_xdot)

            # Constrain slip ratio to be at least a certain magnitude.
            if i!=self.T:
                self.prog.AddConstraint(slip_ratios[i, 0]**2.0 >= self.min_slip_ratio_mag**2.0)
                self.prog.AddConstraint(slip_ratios[i, 1]**2.0 >= self.min_slip_ratio_mag**2.0)

        # Control rate limits.
        for i in range(self.T-1):
            ddelta = self.dt * (steers[i+1] - steers[i])
            f_dkappa = self.dt * (slip_ratios[i+1, 0] - slip_ratios[i, 0])
            r_dkappa = self.dt * (slip_ratios[i+1, 1] - slip_ratios[i, 1])
            self.prog.AddConstraint(ddelta <= self.max_ddelta)
            self.prog.AddConstraint(ddelta >= -self.max_ddelta)
            self.prog.AddConstraint(f_dkappa <= self.max_dkappa)
            self.prog.AddConstraint(f_dkappa >= -self.max_dkappa)
            self.prog.AddConstraint(r_dkappa <= self.max_dkappa)
            self.prog.AddConstraint(r_dkappa >= -self.max_dkappa)
        
        # Control value limits.
        for i in range(self.T):
            self.prog.AddConstraint(steers[i] <= self.max_delta)
            self.prog.AddConstraint(steers[i] >= -self.max_delta)

        # Add dynamics constraints by constraining residuals to be zero.
        for i in range(self.T):
            residuals = self.dynamics.nominal_dynamics(state[i], state[i+1], nominal_forces[i], steers[i])
            for r in residuals:
                self.prog.AddConstraint(r==0.0)

        # Add deterministic force constraints as an initial test.
        self.add_deterministic_force_constraints(state, nominal_forces, steers, self.tire_model.get_combined_slip_model(pmath), slip_ratios)

        # Add the cost function.
        self.add_cost(state)

        # Generate initial guess
        initial_guess = self.constant_input_initial_guess(state, steers, slip_ratios, nominal_forces)
        return initial_guess

    def constant_input_initial_guess(self, state, steers, slip_ratios, nominal_forces):
        # Generate the numpy array for guesses.
        gslip_ratios = np.tile(np.array([self.min_slip_ratio_mag, self.min_slip_ratio_mag]), (self.T,1))
        gsteers = -0.1 * np.ones(self.T) #TODO: hard coded
        gstate = np.empty((self.T + 1, self.n_state))
        gstate[0] = pack_state_vector(self.initial_state)
        gforces = np.empty((self.T, 4))
        pacejka_model = self.tire_model.get_combined_slip_model(math)

        # Simulate the dynamics.
        for i in range(self.T):
            gstate[i+1], forces = self.dynamics.simulate(gstate[i], gsteers[i], gslip_ratios[i, 0], gslip_ratios[i, 1], pacejka_model, self.dt)
            gforces[i] = pack_force_vector(forces)

        psis = gstate[:, 2]
        xs = gstate[:, 4]
        ys = gstate[:, 5]
        fig, ax = plt.subplots()
        plot_planned_trajectory(ax, xs, ys, psis, 2.0, 1.0)
        plt.show()

        # Declare array for the initial guess and set the values.
        initial_guess = np.empty(self.prog.num_vars())
        self.prog.SetDecisionVariableValueInVector(state, gstate, initial_guess)
        self.prog.SetDecisionVariableValueInVector(steers, gsteers, initial_guess)
        self.prog.SetDecisionVariableValueInVector(slip_ratios, gslip_ratios, initial_guess)
        self.prog.SetDecisionVariableValueInVector(nominal_forces, gforces, initial_guess)
        return initial_guess

    def solve_program(self, initial_guess):
        solver = SnoptSolver()
        result = solver.Solve(self.prog, initial_guess)
        solver_details = result.get_solver_details()
        print("Exit flag: " + str(solver_details.info))

        state_res = result.GetSolution(self.state)


    def add_cost(self, state):
        # Add the final state cost function.
        diff_state = pack_state_vector(self.goal_state) - state[-1]
        self.prog.AddQuadraticCost(diff_state.T @ self.Qf @ diff_state)

        # Get the approx distance function for the ellipse.
        fun = self.ellipse.approx_dist_fun()
        for i in range(self.T):
            s = unpack_state_vector(state[i])
            self.prog.AddCost(self.deviation_cost * fun(s["X"], s["Y"]))

    def add_deterministic_force_constraints(self, state, forces, steers, pacejka_model, slip_ratios):
        """
        Args:
            prog:
            state:
            forces:
            steers:
            pacejka_model: function with signature (slip_ratio, slip_angle) using pydrake.math
        """
        for i in range(self.T):
            # Get slip angles and slip ratios.
            s = unpack_state_vector(state[i])
            F = unpack_force_vector(forces[i])
            delta = steers[i]
            alpha_f, alpha_r = self.dynamics.slip_angles(s["xdot"], s["ydot"], s["psidot"], delta)
            kappa_f = slip_ratios[i, 0]
            kappa_r = slip_ratios[i, 1]
            Ffx, Ffy = pacejka_model(kappa_f, alpha_f)
            Frx, Fry = pacejka_model(kappa_r, alpha_r)

            # Constrain the values from the pacejka model to be equal
            # to the desired values in the optimization.
            self.prog.AddConstraint(Ffx - F["f_long"] == 0.0)
            self.prog.AddConstraint(Ffy - F["f_lat"] == 0.0)
            self.prog.AddConstraint(Frx - F["r_long"] == 0.0)
            self.prog.AddConstraint(Fry - F["r_lat"] == 0.0)

if __name__ == "__main__":
    # Configuration variables
    config = {
    "lf" : 0.7,
    "lr" : 0.7,
    # Mass and Iz are for Oldsmobile Ciera 1985: http://www.mchenrysoftware.com/forum/Yaw%20Inertia.pdf
    "m" : 1279.0,
    "Iz" : 2416.97512,
    "T" : 100,
    "dt" : 0.05,
    "pacejka_params" : {"Bx":25, "Cx":2.1, "Dx":30000.0, "Ex":-0.4, 
            "By":15.5, "Cy":2.0, "Dy":20000.0, "Ey":-1.6, "k_alpha_ratio":9.0/7.0},
    "xinit" : {"xdot" : 5.0, "ydot" : 0.0, "psi" : 0.0, "psidot": 0.0, "X" : 0.0, "Y" : 0.0},
    "xgoal" : {"xdot" : 5.0, "ydot" : 0.0, "psi" : -0.5 * math.pi, "psidot": 0.0, "X" : 1.0, "Y" : -1.0},
    "ellipse" : Ellipse(20.0, 20.0, 0, -20.0), # Parameters of the ellipse path
    "deviation_cost" : 1.0, # Cost on deviation from the ellipse
    "Qf" : np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]), # Final state cost matrix.
    "min_xdot" : 0.01, # prevent numerical issues because slip angles divide by xdot
    "min_slip_ratio_mag" : 1e-5, # prevent numerical issues in the Pacejka model
    "max_ddelta" : 0.2, # max rad/s for steering
    "max_dkappa": 0.1, #max change in slip angle per second.
    "max_delta" : 0.2 # maximum steering angle
    }
    opt = Optimization(config)
    initial_guess = opt.build_program()
    opt.solve_program(initial_guess)