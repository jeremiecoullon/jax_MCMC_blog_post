from jax import random
import time
import numpy as onp

from logistic_regression_model import gen_data, build_batch_grad_log_post
from sgld_samplers import run_3_sgld_samplers

"""
Run the 3 ULA samplers while increasing the dataset size
"""

# =============
# These are constant throughout
num_samples = 20000
print_rate = 5000
dim = 5
# =============

# (N,dim) = (1e3, 5)
print("============\nExperiment 1\n============")
key = random.PRNGKey(0)

N = 1000
minibatch_size = int(N*0.1)
dt = 5e-3

theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
onp.savetxt(f"outputs/sgld/increase_data/increase_data_{N}.txt", running_times)

# # ==============
# (N,dim) = (1e4, 5)
print("\n============\nExperiment 2\n============")
key = random.PRNGKey(0)
N = 10000
minibatch_size = int(N*0.1)
dt = 5e-4

theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
onp.savetxt(f"outputs/sgld/increase_data/increase_data_{N}.txt", running_times)

# ==============
# (N,dim) = (1e5, 5)
print("\n============\nExperiment 3\n============")
key = random.PRNGKey(0)
N = 100000
minibatch_size = int(N*0.01)
dt = 5e-5

theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
onp.savetxt(f"outputs/sgld/increase_data/increase_data_{N}.txt", running_times)


# ==============
# (N,dim) = (1mil, 5)
print("\n============\nExperiment 4\n============")
key = random.PRNGKey(0)
N = 1000000
minibatch_size = int(N*0.001)
dt = 5e-6

theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_batch_grad_log_post(X, y_data, N)
running_times = run_3_sgld_samplers(key, grad_log_post, N, dim, dt, num_samples,
                    theta_true, X, y_data, minibatch_size)
onp.savetxt(f"outputs/sgld/increase_data/increase_data_{N}.txt", running_times)
