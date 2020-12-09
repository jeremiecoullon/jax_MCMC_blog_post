from jax import random
import numpy as np
import time

from logistic_regression_model import gen_data, build_grad_log_post
from ula_samplers import run_3_samplers

"""
Run the 3 ULA samplers while increasing the dimension
"""


# These are constant throughout
num_samples = 20000
print_rate = 5000
dt = 5e-3
N = 1000


print("============\nExperiment 1\n============")
key = random.PRNGKey(0)
dim = 5

# gpu
# dim = 100
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_grad_log_post(X, y_data, N)
running_times = run_3_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/ula/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 2\n============")
key = random.PRNGKey(0)
dim = 500

# gpu
# dim = 10000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_grad_log_post(X, y_data, N)
running_times = run_3_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/ula/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 3\n============")
key = random.PRNGKey(0)
dim = 1000

# gpu
# dim = 20000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_grad_log_post(X, y_data, N)
running_times = run_3_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/ula/increase_dimension/dim_{dim}.txt", running_times)


print("\n============\nExperiment 4\n============")
key = random.PRNGKey(0)
dim = 2000

# gpu
# dim = 30000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_grad_log_post(X, y_data, N)
running_times = run_3_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true)
np.savetxt(f"outputs/ula/increase_dimension/dim_{dim}.txt", running_times)
