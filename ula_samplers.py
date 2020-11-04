import jax.numpy as jnp
from jax import partial, jit, vmap, grad, random, lax, ops
import numpy as np
import time

# 1. Pure python/numpy (except for the function `grad_log_post`)
def ula_sampler_python(grad_log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = np.zeros((num_samples, dim))
    paramCurrent = x_0
    print(f"Python sampler:")
    for i in range(num_samples):
        paramGradCurrent = grad_log_post(paramCurrent)
        paramCurrent = paramCurrent + dt*paramGradCurrent + np.sqrt(2*dt)*np.random.normal(size=(paramCurrent.shape))
        samples[i] = paramCurrent
        if i%print_rate==0:
            print(f"Iteration {i}/{num_samples}")
    return samples


# 2. Python loop with Jax kernel
@partial(jit, static_argnums=(2,3))
def ula_kernel(key, param, grad_log_post, dt):
    key, subkey = random.split(key)
    paramGrad = grad_log_post(param)
    param = param + dt*paramGrad + jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    return key, param


def ula_sampler_jax_kernel(key, grad_log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = np.zeros((num_samples, dim))
    param = x_0
    print(f"Python loop with Jax kernel")
    for i in range(num_samples):
        key, param = ula_kernel(key, param, grad_log_post, dt)
        samples[i] = param
        if i%print_rate==0:
            print(f"Iteration {i}/{num_samples}")
    return samples


# 3. Pure Jax
@partial(jit, static_argnums=(1,2,3))
def ula_sampler_full_jax_jit(key, grad_log_post, num_samples, dt, x_0):

    def ula_step(carry, x):
        key, param = carry
        key, param = ula_kernel(key, param, grad_log_post, dt)
        return (key, param), param

    carry = (key, x_0)
    _, samples = lax.scan(ula_step, carry, None, num_samples)
    return samples


def run_3_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true):
    """
    Run and time the 3 samplers
    """
    print_rate = num_samples/2

    print(f"Time the 3 samplers with N={N}, dim={dim}\n")
    start = time.time()
    ula_samples_python = ula_sampler_python(grad_log_post, num_samples, dt=dt,
                                            x_0=theta_true, print_rate=print_rate)

    end = time.time()
    running_time_1 = end-start
    lemin = int(running_time_1/60)
    lesec = int(running_time_1 % 60)
    print(f"Python sampler: {running_time_1:.1f}sec ({lemin}min {lesec}sec)\n")

    subkey1, subkey2 = random.split(key)
    start = time.time()
    ula_samples_jax1 = ula_sampler_jax_kernel(subkey1, grad_log_post, num_samples,
                                              dt=dt, x_0=theta_true, print_rate=print_rate)
    end = time.time()
    running_time_2 = end-start
    lemin = int(running_time_2/60)
    lesec = int(running_time_2 % 60)
    print(f"Jax kernel sampler: {(end-start):.2f}sec ({lemin}min {lesec}sec)\n")

    start = time.time()
    ula_samples_jax_full = ula_sampler_full_jax_jit(subkey2, grad_log_post, num_samples,
                                                       dt, theta_true)
    _ = ula_samples_jax_full[0][0].block_until_ready()
    end = time.time()
    running_time_3 = end-start
    print(f"Pure Jax sampler, 1st run: {(end-start):.3f}sec\n")

    start = time.time()
    ula_samples_jax_full = ula_sampler_full_jax_jit(subkey2, grad_log_post, num_samples,
                                                       dt, theta_true)
    _ = ula_samples_jax_full[0][0].block_until_ready()
    end = time.time()
    running_time_4 = end-start
    print(f"Pure Jax sampler, 2nd run: {(end-start):.3f}sec\n")
    print("Done.")
    return [running_time_1, running_time_2, running_time_3, running_time_4]
