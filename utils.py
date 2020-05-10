import numpy as np

def array_norm(array1, array2):
  """
  Elementwise norm of two arrays.
  """
  return np.sqrt(np.square(array1) + np.square(array2))

def unpack_state_vector(state):
  """
  Given a 1d array of state, return a dictionary mapping it to 
  state variables
  """
  state_dic = {
               "xdot" : state[0], 
               "ydot" : state[1], 
               "psi" : state[2],
               "psidot" : state[3],
               "X" : state[4], 
               "Y" : state[5] 
               }
  return state_dic

def pack_state_vector(s_dic):
  """
  Given a state dictionary, return a 1d state vector.
  """
  state = np.array([s_dic["xdot"],
                   s_dic["ydot"],
                   s_dic["psi"],
                   s_dic["psidot"],
                   s_dic["X"],
                   s_dic["Y"]])
  return state

def unpack_force_vector(force):
  force_dic = {
      "f_long" : force[0],
      "f_lat" : force[1],
      "r_long" : force[2],
      "r_lat" : force[3]
  }
  return force_dic

def pack_force_vector(f_dic):
  fvec = np.array([
    f_dic["f_long"],
    f_dic["f_lat"],
    f_dic["r_long"],
    f_dic["r_lat"]
  ])
  return fvec

def extract_time_series(state):
  """ Given the full state array, extract the time series.
  Args:
      state (T + 1 by 6 numpy array): [description]
  """
  states = {"xdot" : state[:, 0].tolist(),
            "ydot" : state[:, 1].tolist(),
            "psi" : state[:, 2].tolist(),
            "psidot" : state[:, 3].tolist(),
            "X" : state[:, 4].tolist(),
            "Y" : state[:, 5].tolist()}
  return states