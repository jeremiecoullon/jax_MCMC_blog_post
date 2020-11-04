from jax import random
import numpy as np
import time

from logistic_regression_model import gen_data, build_value_and_grad_log_post
from mala_samplers import run_3_mala_samplers

"""
Run the 3 MALA samplers while increasing the dimension
"""


# These are constant throughout
num_samples = 20000
print_rate = 5000
dt = 5e-3
N = 1000


print("============\nExperiment 1\n============")
key = random.PRNGKey(0)
dim = 5
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 2\n============")
key = random.PRNGKey(0)
dim = 500
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 3\n============")
key = random.PRNGKey(0)
dim = 1000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 4\n============")
key = random.PRNGKey(0)
dim = 2000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_dimension/dim_{dim}.txt", running_times)
