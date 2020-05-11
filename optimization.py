import numpy as np
import math
from pydrake.all import MathematicalProgram, SnoptSolver, IpoptSolver, SolverOptions
import pydrake.math as pmath
import matplotlib.pyplot as plt

from dynamics import Dynamics
from tire_model import TireModel
from utils import unpack_force_vector, unpack_state_vector, pack_state_vector, pack_force_vector, extract_time_series
from world import EllipseArc
from visualize import plot_planned_trajectory, plot_slips, plot_puddles, plot_forces, plot_ellipse_arc
from animate import generate_animation
from puddles import PuddleModel
from config import *

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
        self.ellipse_arc = config["ellipse_arc"]
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
        self.initial_guess_config = config["initial_guess_config"]
        self.puddle_model = config["puddle_model"]
        self.force_constraint = config["force_constraint"]
        self.visualize_initial_guess = config["visualize_initial_guess"]
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

        # Add the cost function.
        self.add_cost(state)

        # Add a different force constraint depending on the configuration.
        if self.force_constraint == ForceConstraint.NO_PUDDLE:
            self.add_no_puddle_force_constraints(state, nominal_forces, steers, self.tire_model.get_deterministic_model(), slip_ratios)
        elif self.force_constraint == ForceConstraint.MEAN_CONSTRAINED:
            self.add_mean_constrained()
        else:
            raise NotImplementedError("ForceConstraint type not implemented.")
        return

    def constant_input_initial_guess(self, state, steers, slip_ratios, nominal_forces):
        # Guess the slip ratio as the minimum allowed value.
        gslip_ratios = np.tile(np.array([self.min_slip_ratio_mag, self.min_slip_ratio_mag]), (self.T,1))

        # Guess the slip angle as a linearly ramping steer that then becomes constant.
        # This is done by creating an array of values corresponding to end_delta then
        # filling in the initial ramp up phase.
        gsteers = self.initial_guess_config["end_delta"] * np.ones(self.T)
        igc = self.initial_guess_config
        for i in range(igc["ramp_steps"]):
            gsteers[i] = (i/igc["ramp_steps"]) * (igc["end_delta"] - igc["start_delta"])

        # Create empty arrays for state and forces.
        gstate = np.empty((self.T + 1, self.n_state))
        gstate[0] = pack_state_vector(self.initial_state)
        gforces = np.empty((self.T, 4))
        all_slips = self.T * [None]

        if self.force_constraint == ForceConstraint.NO_PUDDLE:
            tire_model = self.tire_model.get_deterministic_model()
            for i in range(self.T):
                s = unpack_state_vector(gstate[i])

                # Simulate the dynamics for one time step.
                gstate[i+1], forces, slips = self.dynamics.simulate(gstate[i], gsteers[i], gslip_ratios[i, 0], gslip_ratios[i, 1], tire_model, self.dt)
                
                # Store the results.
                gforces[i] = pack_force_vector(forces)
                all_slips[i] = slips

        elif self.force_constraint == ForceConstraint.MEAN_CONSTRAINED or self.force_constraint==ForceConstraint.CHANCE_CONSTRAINED:
            # mean model is a function that maps (slip_ratio, slip_angle, x, y) -> (E[Fx], E[Fy])
            mean_model = self.tire_model.get_mean_model(self.puddle_model.get_mean_fun())

            for i in range(self.T):
                # Update the tire model based off the conditions of the world
                # at (x, y)
                s = unpack_state_vector(gstate[i])
                tire_model = lambda slip_ratio, slip_angle : mean_model(slip_ratio, slip_angle, s["X"], s["Y"])

                # Simulate the dynamics for one time step.
                gstate[i+1], forces, slips = self.dynamics.simulate(gstate[i], gsteers[i], gslip_ratios[i, 0], gslip_ratios[i, 1], tire_model, self.dt)
                
                # Store the results.
                gforces[i] = pack_force_vector(forces)
                all_slips[i] = slips

        # Declare array for the initial guess and set the values.
        initial_guess = np.empty(self.prog.num_vars())
        self.prog.SetDecisionVariableValueInVector(state, gstate, initial_guess)
        self.prog.SetDecisionVariableValueInVector(steers, gsteers, initial_guess)
        self.prog.SetDecisionVariableValueInVector(slip_ratios, gslip_ratios, initial_guess)
        self.prog.SetDecisionVariableValueInVector(nominal_forces, gforces, initial_guess)

        if self.visualize_initial_guess:
            # TODO: reorganize visualizations
            psis = gstate[:, 2]
            xs = gstate[:, 4]
            ys = gstate[:, 5]
            fig, ax = plt.subplots()
            plot_puddles(ax, self.puddle_model)
            plot_ellipse_arc(ax, self.ellipse_arc)
            plot_planned_trajectory(ax, xs, ys, psis, gsteers)
            # Plot the slip ratios/angles
            plot_slips(all_slips)
            plot_forces(gforces)
            generate_animation(xs, ys, psis, gsteers, self.lf, self.lr, 0.5, 0.25, self.dt, puddle_model=self.puddle_model)

        return initial_guess

    def solve_program(self):
        # Generate initial guess
        initial_guess = self.constant_input_initial_guess(self.state, self.steers, self.slip_ratios, self.nominal_forces)

        # Solve the problem.
        solver = SnoptSolver()
        result = solver.Solve(self.prog, initial_guess)
        solver_details = result.get_solver_details()
        print("Exit flag: " + str(solver_details.info))

        self.visualize_results(result)


    def visualize_results(self, result):
        state_res = result.GetSolution(self.state)
        psis = state_res[:, 2]
        xs = state_res[:, 4]
        ys = state_res[:, 5]
        steers = result.GetSolution(self.steers)

        fig, ax = plt.subplots()
        plot_puddles(ax, self.puddle_model)
        plot_planned_trajectory(ax, xs, ys, psis, steers)
        generate_animation(xs, ys, psis, steers, self.lf, self.lr, 0.5, 0.25, self.dt, puddle_model=self.puddle_model)
        plt.show()

    def add_cost(self, state):
        # Add the final state cost function.
        diff_state = pack_state_vector(self.goal_state) - state[-1]
        self.prog.AddQuadraticCost(diff_state.T @ self.Qf @ diff_state)

        # Get the approx distance function for the ellipse.
        fun = self.ellipse_arc.approx_dist_fun()
        for i in range(self.T):
            s = unpack_state_vector(state[i])
            self.prog.AddCost(self.deviation_cost * fun(s["X"], s["Y"]))

    def add_mean_constrained(self):
        # mean model is a function that maps (slip_ratio, slip_angle, x, y) -> (E[Fx], E[Fy])
        mean_model = self.tire_model.get_mean_model(self.puddle_model.get_mean_fun())

        for i in range(self.T):
            # Get slip angles and slip ratios.
            s = unpack_state_vector(self.state[i])
            F = unpack_force_vector(self.nominal_forces[i])
            # get the tire model at this position in space.
            tire_model = lambda slip_ratio, slip_angle : mean_model(slip_ratio, slip_angle, s["X"], s["Y"])

            # Unpack values
            delta = self.steers[i]
            alpha_f, alpha_r = self.dynamics.slip_angles(s["xdot"], s["ydot"], s["psidot"], delta)
            kappa_f = self.slip_ratios[i, 0]
            kappa_r = self.slip_ratios[i, 1]

            # Compute expected forces.
            E_Ffx, E_Ffy = tire_model(kappa_f, alpha_f)
            E_Frx, E_Fry = tire_model(kappa_r, alpha_r)

            # Constrain these expected force values to be equal to the nominal
            # forces in the optimization.
            self.prog.AddConstraint(E_Ffx - F["f_long"] == 0.0)
            self.prog.AddConstraint(E_Ffy - F["f_lat"] == 0.0)
            self.prog.AddConstraint(E_Frx - F["r_long"] == 0.0)
            self.prog.AddConstraint(E_Fry - F["r_lat"] == 0.0)

    def add_no_puddle_force_constraints(self, state, forces, steers, pacejka_model, slip_ratios):
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
    initial_guess_config = {"start_delta" : 0.0, "end_delta" : 0.5, "ramp_steps" : 50}

    # Create a puddle model.
    centers = [np.array([4.0, 3.0])]
    shapes = [np.array([[5.0, -1.0], [-1.0, 6.0]])]
    mean_scales = [1.0]
    variance_scales = [0.5] 
    puddle_model = PuddleModel(centers, shapes, mean_scales, variance_scales)


    # Configuration variables
    config = {
    "lf" : 0.7,
    "lr" : 0.7,
    # Mass and Iz are for Oldsmobile Ciera 1985: http://www.mchenrysoftware.com/forum/Yaw%20Inertia.pdf
    "m" : 1279.0,
    "Iz" : 2416.97512,
    "T" : 200,
    "dt" : 0.01,
    "pacejka_params" : {"Bx":25, "Cx":2.1, "Dx":30000.0, "Ex":-0.4, 
            "By":15.5, "Cy":2.0, "Dy":20000.0, "Ey":-1.6, "k_alpha_ratio":9.0/7.0},
    "xinit" : {"xdot" : 5.0, "ydot" : 0.0, "psi" : 0.0, "psidot": 0.0, "X" : 0.0, "Y" : 0.0},
    "xgoal" : {"xdot" : 5.0, "ydot" : 0.0, "psi" : math.pi, "psidot": 0.0, "X" : 0.0, "Y" : 6.0},
    "ellipse_arc" : EllipseArc(5.0, 2.5, 0, 2.5, -0.5*math.pi, 0.5*math.pi), # Parameters of the ellipse path
    "deviation_cost" : 1.0, # Cost on deviation from the ellipse
    "Qf" : np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]), # Final state cost matrix.
    "min_xdot" : 0.01, # prevent numerical issues because slip angles divide by xdot
    "min_slip_ratio_mag" : 1e-5, # prevent numerical issues in the Pacejka model
    "max_ddelta" : 0.2, # max rad/s for steering
    "max_dkappa": 0.1, #max change in slip angle per second.
    "max_delta" : 0.2, # maximum steering angle
    "initial_guess_config" : initial_guess_config, # parameters for generating the initial guess\
    "puddle_model" : puddle_model, # an instance of PuddleModel
    "force_constraint" : ForceConstraint.MEAN_CONSTRAINED, # specify the type of force constraint
    "visualize_initial_guess" : True
    }
    opt = Optimization(config)
    opt.build_program()
    opt.solve_program()