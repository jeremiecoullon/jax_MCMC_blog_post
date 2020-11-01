import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from jax import random
import time

from logistic_regression_model import gen_data, build_grad_log_post
from samplers import ula_sampler_python

"""
Run pure python sampler for (N, dim) = (1000, 4) and plot the samples
"""

num_samples = 20000
print_rate = 5000

# ======
# build dataset and grad_log_post
key = random.PRNGKey(0)
N = 1000
dim = 4
dt = 5e-3

theta_true, X, y_data = gen_data(key, dim, N)
grad_log_post = build_grad_log_post(X, y_data, N)
print("====")
start = time.time()
ula_samples_python = ula_sampler_python(grad_log_post, num_samples, dt=dt, x_0=theta_true, print_rate=print_rate)

end = time.time()
print(f"Running time: {(end-start):.1f}sec")



dim_list = [0,1,2,3]
fig, ax = plt.subplots(1, len(dim_list), figsize=(14,3))

for idx, k in enumerate(dim_list):
    ax[idx].plot(ula_samples_python[:,k])
    ax[idx].axhline(theta_true[k],c='r')
    ax[idx].set_title(f"theta {k}", size=17)
    ax[idx].set_xlabel("Iterations", size=17)

plt.tight_layout()
plt.show()
