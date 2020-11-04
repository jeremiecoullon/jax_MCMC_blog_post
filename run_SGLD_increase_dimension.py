from jax import random
import numpy as np
import time

from logistic_regression_model import gen_data, build_batch_grad_log_post
from sgld_samplers import run_3_sgld_samplers

"""
Run the 3 SGLD samplers while increasing the dimension
"""


# These are constant throughout
num_samples = 20000
print_rate = 5000
dt = 5e-3
N = 1000
minibatch_size = int(N*0.1)


print("============\nExperiment 1\n============")
key = random.PRNGKey(0)
dim = 5
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
np.savetxt(f"outputs/sgld/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 2\n============")
key = random.PRNGKey(0)
dim = 500
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
np.savetxt(f"outputs/sgld/increase_dimension/dim_{dim}.txt", running_times)

# # ==============
print("\n============\nExperiment 3\n============")
key = random.PRNGKey(0)
dim = 1000
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
np.savetxt(f"outputs/sgld/increase_dimension/dim_{dim}.txt", running_times)

# ==============
print("\n============\nExperiment 4\n============")
key = random.PRNGKey(0)
dim = 2000
print("Generating data (this take a while)..")
theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
np.savetxt(f"outputs/sgld/increase_dimension/dim_{dim}.txt", running_times)
