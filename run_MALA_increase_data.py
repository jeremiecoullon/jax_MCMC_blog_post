from jax import random
import time
import numpy as np

from logistic_regression_model import gen_data, build_value_and_grad_log_post
from mala_samplers import run_3_mala_samplers

"""
Run the 3 MALA samplers while increasing the dataset size
"""

# =============
# These are constant throughout
num_samples = 20000
print_rate = 5000
dim = 5
# =============


print("============\nExperiment 1\n============")
key = random.PRNGKey(0)

N = 1000
dt = 5e-3
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_data/increase_data_{N}.txt", running_times)

# ==============
print("\n============\nExperiment 2\n============")
key = random.PRNGKey(0)
N = 10000
dt = 5e-4

# gpu
# N = 1000000
# dt = 5e-6
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_data/increase_data_{N}.txt", running_times)

# ==============
print("\n============\nExperiment 3\n============")
key = random.PRNGKey(0)
N = 100000
dt = 5e-5

# gpu
# N = 10000000
# dt = 5e-7
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_data/increase_data_{N}.txt", running_times)


# ==============
print("\n============\nExperiment 4\n============")
key = random.PRNGKey(0)
N = 1000000
dt = 5e-6

# gpu
# N = 20000000 
# dt = 5e-8
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_value_and_grad_log_post(X, y_data, N)
running_times = run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/mala/increase_data/increase_data_{N}.txt", running_times)
