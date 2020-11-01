# -*- coding: utf-8 -*-

import numpy as onp
import jax.numpy as np
from jax import random, partial, jit, lax
import time


# 1. Numpy version
def mala_sampler_python(log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = onp.zeros((num_samples, dim))
    paramCurrent = x_0
    accepts = 0
    current_log_post, current_grad = log_post(paramCurrent)

    start = time.time()
    print(f"Running MALA for {num_samples} iterations with dt={dt}")
    for i in range(num_samples):
        paramProp = paramCurrent + dt*current_grad + onp.sqrt(2*dt)*onp.random.normal(size=dim)
        new_log_post, new_grad = log_post(paramProp)

        term1 = paramProp - paramCurrent - dt*current_grad
        term2 = paramCurrent - paramProp - dt*new_grad
        q_new = -0.25*(1/dt)*onp.dot(term1, term1)
        q_current = -0.25*(1/dt)*onp.dot(term2, term2)

        log_ratio = new_log_post - current_log_post + q_current - q_new
        if onp.log(onp.random.uniform()) < log_ratio:
            paramCurrent = paramProp
            current_log_post = new_log_post
            current_grad = new_grad
            accepts += 1
        samples[i] = paramCurrent
        if i%print_rate==0:
            print(f"Iteration {i}/{num_samples}")

    end = time.time()
    print("Done.")
    print(f"Running time: {(end-start):.2f}sec")
    accept_rate = accepts/num_samples * 100
    print(f"Acceptance rate: {accept_rate:.1f}%")

    return samples


# 2. Python loop with kernel
@partial(jit, static_argnums=(3,5))
def mala_kernel(key, paramCurrent, paramGradCurrent, log_post, logpostCurrent, dt):
    key, subkey1, subkey2 = random.split(key, 3)
    paramProp = paramCurrent + dt*paramGradCurrent + np.sqrt(2*dt)*random.normal(key=subkey1, shape=paramCurrent.shape)
    new_log_post, new_grad = log_post(paramProp)

    term1 = paramProp - paramCurrent - dt*paramGradCurrent
    term2 = paramCurrent - paramProp - dt*new_grad
    q_new = -0.25*(1/dt)*np.dot(term1, term1)
    q_current = -0.25*(1/dt)*np.dot(term2, term2)

    log_ratio = new_log_post - logpostCurrent + q_current - q_new
    acceptBool = np.log(random.uniform(key=subkey2)) < log_ratio
    paramCurrent = np.where(acceptBool, paramProp, paramCurrent)
    current_grad = np.where(acceptBool, new_grad, paramGradCurrent)
    current_log_post = np.where(acceptBool, new_log_post, logpostCurrent)
    accepts_add = np.where(acceptBool, 1,0)
    return key, paramCurrent, current_grad, current_log_post, accepts_add


def mala_sampler_jax_kernel(key, log_post, num_samples, dt, x_0, print_rate=500):
    dim, = x_0.shape
    samples = onp.zeros((num_samples, dim))
    paramCurrent = x_0
    accepts = 0
    current_log_post, current_grad = log_post(paramCurrent)

    start = time.time()
    print(f"Running MALA for {num_samples} iterations with dt={dt}")
    for i in range(num_samples):
        key, paramCurrent, current_grad, current_log_post, accepts_add = mala_kernel(key,
                                                                        paramCurrent,
                                                                        current_grad,
                                                                        log_post,
                                                                        current_log_post,
                                                                        dt)
        accepts += accepts_add
        samples[i] = paramCurrent
        if i%print_rate==0:
            print(f"Iteration {i}/{num_samples}")

    end = time.time()
    print("Done.")
    print(f"Running time: {(end-start):.2f}sec")
    accept_rate = accepts/num_samples * 100
    print(f"Acceptance rate: {accept_rate:.1f}%")

    return samples



@partial(jit, static_argnums=(1,2,3))
def mala_sampler_full_jax(key, val_and_grad_log_post, num_samples, dt, x_0):

    def mala_step(carry, x):
        key, paramCurrent, gradCurrent, logpostCurrent, accepts = carry
        key, paramCurrent, gradCurrent, logpostCurrent, accepts_add = mala_kernel(key, paramCurrent, gradCurrent, val_and_grad_log_post, logpostCurrent, dt)
        accepts += accepts_add
        return (key, paramCurrent, gradCurrent, logpostCurrent, accepts), paramCurrent

    paramCurrent = x_0
    logpostCurrent, gradCurrent = val_and_grad_log_post(x_0)
    carry = (key, paramCurrent, gradCurrent, logpostCurrent, 0)
    (_, _, _, _, accepts), samples = lax.scan(mala_step, carry, None, num_samples)
    return samples, 100*(accepts/num_samples)


def run_3_mala_samplers(key, grad_log_post, N, dim, dt, num_samples, theta_true):
    """
    Run and time the 3 MALA samplers
    """
    print_rate = num_samples/2

    print(f"Time the 3 samplers with N={N}, dim={dim}\n")
    start = time.time()
    mala_samples_python = mala_sampler_python(grad_log_post, num_samples, dt=dt,
                                            x_0=theta_true, print_rate=print_rate)

    end = time.time()
    running_time_1 = end-start
    lemin = int(running_time_1/60)
    lesec = int(running_time_1 % 60)
    print(f"Python sampler: {running_time_1:.1f}sec ({lemin}min {lesec}sec)\n")

    subkey1, subkey2 = random.split(key)
    start = time.time()
    mala_samples_jax1 = mala_sampler_jax_kernel(subkey1, grad_log_post, num_samples,
                                              dt=dt, x_0=theta_true, print_rate=print_rate)
    end = time.time()
    running_time_2 = end-start
    lemin = int(running_time_2/60)
    lesec = int(running_time_2 % 60)
    print(f"Jax kernel sampler: {(end-start):.2f}sec ({lemin}min {lesec}sec)\n")

    start = time.time()
    mala_samples_jax_full = mala_sampler_full_jax(subkey2, grad_log_post, num_samples,
                                                       dt, theta_true)
    _ = mala_samples_jax_full[0][0].block_until_ready()
    end = time.time()
    running_time_3 = end-start
    print(f"Pure Jax sampler, 1st run: {(end-start):.3f}sec\n")

    start = time.time()
    mala_samples_jax_full = mala_sampler_full_jax(subkey2, grad_log_post, num_samples,
                                                       dt, theta_true)
    _ = mala_samples_jax_full[0][0].block_until_ready()
    end = time.time()
    running_time_4 = end-start
    print(f"Pure Jax sampler, 2nd run: {(end-start):.3f}sec\n")
    print("Done.")
    return [running_time_1, running_time_2, running_time_3, running_time_4]
