import jax
from jax import random, jit, numpy as jnp
from einops import repeat, pack, unpack, rearrange
import math
from jax_tqdm import loop_tqdm
from tqdm import tqdm
from functools import partial

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    #dtype = x.dtype
    #x, y = x.astype(jnp.float64).double(), y.double()
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    unit = y / norm
    #unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(axis = -1, keepdims = True) * unit
    orthogonal = x - parallel

    return inverse(parallel), inverse(orthogonal)

def logsnr_schedule_linear(t: jax.Array, clip_min = 1e-9):
    alpha = jnp.clip(1 - t, clip_min, 1.)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared / sigma_squared)

def logsnr_schedule_sigmoid(t: jax.Array, start=0, end=3, tau=.5):
    v_start = jax.nn.sigmoid(start / tau)
    v_end = jax.nn.sigmoid(end / tau)
    output = jax.nn.sigmoid(t * ((end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    alpha = jnp.clip(output, 1e-9, 1)
    alpha_squared = alpha ** 2
    sigma_squared = 1 - alpha_squared
    return jnp.log(alpha_squared/sigma_squared)

def logsnr_schedule_cosine_interpolated(
        t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15, shift: int =128
) -> jax.Array:
    return t * logsnr_schedule_cosine(t=t, shift=shift, reference=256) * (1-t) * logsnr_schedule_cosine(t=t,shift=shift, reference=32)

def logsnr_schedule_cosine(
    t: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15, shift: int =128, reference: int=32
) -> jax.Array:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min))) + 2 * jnp.log(reference / shift)

def logsnr_schedule_laplace(t: jax.Array, mu = 0.0, b = .01):
    return mu - b * jnp.sign(0.5 - t) * jnp.log(1 - 2 * jnp.abs(t - 0.5))

def inv_logsnr_schedule_cosine(lam: jax.Array, logsnr_min: float = -15, logsnr_max: float = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return (jnp.arctan(jnp.exp(-lam*0.5)) - t_min) / (t_max - t_min)

def right_pad_dims_to(x: jax.Array, t: jax.Array) -> jax.Array:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))

def q_sample(x_start: jax.Array, times: jax.Array, noise: jax.Array, schedule=logsnr_schedule_cosine) -> tuple[jax.Array, jax.Array]:
    log_snr = schedule(times)
    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
        jax.nn.sigmoid(-log_snr_padded)
    )
    x_noised = x_start * alpha + noise * sigma
    return x_noised, log_snr


def p_sample_with_condition_scale(module, params, key, x, time, time_next, objective="v", condition = None, condition_scale = 1.0, schedule=logsnr_schedule_cosine, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    null_logits = module.apply(params, x, time=batch_log_snr, condition = jnp.zeros_like(condition), **kwargs)
    logits = module.apply(params, x, time=batch_log_snr, condition = condition, **kwargs)
    update = logits - null_logits

    # https://arxiv.org/abs/2410.02416 -> parallel component primarily causes oversaturation
    keep_parallel_frac = 0.
    parallel, orthog = project(update, logits)
    update = orthog + parallel * keep_parallel_frac

    pred = logits + update * (condition_scale - 1.)


    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c

    noise = random.normal(key, x.shape)
    return jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + jnp.sqrt(posterior_variance) * noise,
        model_mean,
    )

def p_sample_with_mask(module, params, key, x, time, time_next, objective="v", mask=None, gt=None, schedule=logsnr_schedule_cosine, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    
    gt_noise_key, step_noise_key = random.split(key)

    # Sample noised ground truth for timestep
    gt_noise = random.normal(gt_noise_key, gt.shape)
    noised_gt, _ = q_sample(gt, time, gt_noise, schedule=schedule) 
    x = noised_gt * mask + (1-mask) * x
    
    # Denoise step
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c

    noise = random.normal(step_noise_key, x.shape)
    prediction = jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + jnp.sqrt(posterior_variance) * noise,
        model_mean,
    )
    return jax.lax.cond(time_next == 0, lambda m: (mask * gt) + (1-mask) * m, lambda m: m, prediction)

def p_sample(module, params, key, x, time, time_next, objective="v", schedule=logsnr_schedule_cosine, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
    posterior_variance = squared_sigma_next * c

    noise = random.normal(key, x.shape)
    return jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + jnp.sqrt(posterior_variance) * noise,
        model_mean,
    )


def simple_p_sample(module, params, key, x, time, time_next, objective="v", schedule=logsnr_schedule_cosine, noise_param=0.3, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)
    #x_start = dynamic_threshold(x_start)

    #t=current
    #s=next

    #alpha_st = alpha_next / alpha
    #mu = jnp.exp(log_snr - log_snr_next) * alpha_st * x + (1 - jnp.exp(log_snr - log_snr_next)) * alpha_next * x_start
    #min_lvar = jnp.log(1 - jnp.exp(log_snr - log_snr_next)) + jax.nn.log_sigmoid(-log_snr_next)
    #max_lvar = jnp.log(1 - jnp.exp(log_snr - log_snr_next)) + jax.nn.log_sigmoid(-log_snr)

    alpha_st = alpha_next / alpha
    mu = jnp.exp(log_snr - log_snr_next) * alpha_st * x + c * alpha_next * x_start
    min_lvar = jnp.log(c) + jax.nn.log_sigmoid(-log_snr_next)
    max_lvar = jnp.log(c) + jax.nn.log_sigmoid(-log_snr)
    sigma = jnp.sqrt(jnp.exp(noise_param * max_lvar + (1 - noise_param) * min_lvar))

    # sigma_{t->s}^2
    #sigma = jnp.sqrt(squared_sigma_next * c)

    noise = random.normal(key, x.shape)
    return jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + sigma * noise,
        mu,
    )


def simple_repaint(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
    steps = jnp.linspace(0.0, 1.0, num_sample_steps + 1)

    t_T = num_sample_steps + 1
    jump_len = 10
    jump_n_sample = 10
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t-1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t=t+1
                ts.append(t)

    #print("ts:", ts)
    n = len(ts)-1
    #print("n:", n)
    ts = jnp.array(ts)

    #print(steps[ts])

    keys = random.split(key, n)

    for i in tqdm(range(n)):
        index = ts[i]
        index_next = ts[i+1]

        time = steps[index]
        time_next = steps[index_next]

        #print("index:", index, "next:", index_next)
        #print("time:", time, "next:", time_next)

        if time > time_next:
            #print("denoising")
            img = simple_p_sample_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
        else:
            #print("noising")
            k2, k3 = random.split(keys[i])

            # Forward noising
            # Perform markov transition to previous timestep
            noise = random.normal(k2, img.shape)
            log_snr = logsnr_schedule_cosine(time)
            log_snr_next = logsnr_schedule_cosine(time_next)
            log_snr_padded = right_pad_dims_to(img, log_snr)
            log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
            alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
                jax.nn.sigmoid(-log_snr_padded)
            )
            alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
                jax.nn.sigmoid(-log_snr_next_padded)
            )

            alpha_t = alpha_next
            alpha_s = alpha
            sigma_t = sigma_next
            sigma_s = sigma
            zs = img

            alpha_ts = alpha_t / alpha_s
            sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
            sigma_ts = jnp.sqrt(sigma_ts_sq)
            img = zs * alpha_ts + noise * sigma_ts
        #print("new img:", img.min(), img.max(), img.mean(), img.shape)

    #img = jax.lax.fori_loop(0, n, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def simple_p_sample_with_mask(module, params, key, x, time, time_next, objective="v", mask=None, gt=None, schedule=logsnr_schedule_cosine, noise_param=0.3, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    
    gt_noise_key, step_noise_key = random.split(key)

    # Sample noised ground truth for timestep
    gt_noise = random.normal(gt_noise_key, gt.shape)
    noised_gt, _ = q_sample(gt, time, gt_noise, schedule=schedule) 
    x = noised_gt * mask + (1-mask) * x
    
    # Denoise step
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)
    #x_start = dynamic_threshold(x_start)

    alpha_st = alpha_next / alpha
    mu = jnp.exp(log_snr - log_snr_next) * alpha_st * x + c * alpha_next * x_start
    min_lvar = jnp.log(c) + jax.nn.log_sigmoid(-log_snr_next)
    max_lvar = jnp.log(c) + jax.nn.log_sigmoid(-log_snr)
    sigma = jnp.sqrt(jnp.exp(noise_param * max_lvar + (1 - noise_param) * min_lvar))

    noise = random.normal(step_noise_key, x.shape)
    prediction = jax.lax.cond(
        time_next == 0,
        lambda m: m,
        lambda m: m + noise * sigma,
        mu,
    )
    return jax.lax.cond(time_next == 0, lambda m: (mask * gt) + (1-mask) * m, lambda m: m, prediction)

def simple_ddpm_sample_with_mask(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return simple_p_sample_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def _ddpm_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


@partial(jit, static_argnums=(0, 4, 5))
def ddpm_sample_with_condition_scale(module, params, key, img, num_sample_steps=100, objective="v", condition=None, condition_scale=1.0, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample_with_condition_scale(module, params, keys[i], img, time, time_next, objective=objective, condition=condition, condition_scale=condition_scale,**kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def simple_ddpm_sample(module, params, key, img, num_sample_steps=100, objective="v", **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return simple_p_sample(module, params, keys[i], img, time, time_next, objective=objective, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

#@partial(jit, static_argnums=(0, 4, 5))
def ddpm_sample(module, params, key, img, num_sample_steps=100, objective="v", **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample(module, params, keys[i], img, time, time_next, objective=objective, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

#@partial(jit, static_argnums=(0, 4, 5))
#def ddpm_sample_with_mask(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
#    steps = jnp.linspace(0.0, 1.0, num_sample_steps + 1)
#
#    t_T = num_sample_steps + 1
#    jump_len = 4
#    jump_n_sample = 4
#    jumps = {}
#    for j in range(0, t_T - jump_len, jump_len):
#        jumps[j] = jump_n_sample - 1
#    t = t_T
#    ts = []
#    while t >= 1:
#        t = t-1
#        ts.append(t)
#        if jumps.get(t, 0) > 0:
#            jumps[t] = jumps[t] - 1
#            for _ in range(jump_len):
#                t=t+1
#                ts.append(t)
#
#    print("ts:", ts)
#    n = len(ts)-1
#    print("n:", n)
#    ts = jnp.array(ts)
#
#    print(steps[ts])
#
#    keys = random.split(key, n)
#
#    for i in tqdm(range(n)):
#        index = ts[i]
#        index_next = ts[i+1]
#
#        time = steps[index]
#        time_next = steps[index_next]
#
#        print("index:", index, "next:", index_next)
#        print("time:", time, "next:", time_next)
#
#        if time > time_next:
#            print("denoising")
#            img = p_sample_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#        else:
#            print("noising")
#            k2, k3 = random.split(keys[i])
#
#            # Forward noising
#            # Perform markov transition to previous timestep
#            noise = random.normal(k2, img.shape)
#            log_snr = logsnr_schedule_cosine(time)
#            log_snr_next = logsnr_schedule_cosine(time_next)
#            log_snr_padded = right_pad_dims_to(img, log_snr)
#            log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
#            alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
#                jax.nn.sigmoid(-log_snr_padded)
#            )
#            alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
#                jax.nn.sigmoid(-log_snr_next_padded)
#            )
#
#            alpha_t = alpha_next
#            alpha_s = alpha
#            sigma_t = sigma_next
#            sigma_s = sigma
#            zs = img
#
#            alpha_ts = alpha_t / alpha_s
#            sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
#            sigma_ts = jnp.sqrt(sigma_ts_sq)
#            img = zs * alpha_ts + noise * sigma_ts
#        print("new img:", img.min(), img.max(), img.mean(), img.shape)
#
#    #img = jax.lax.fori_loop(0, n, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img

#@partial(jit, static_argnums=(0, 4, 5))
#def ddpm_sample_with_mask(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        #return p_sample_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#
#        k1, aux = random.split(keys[i])
#        new_img = p_sample_with_mask(module, params, k1, img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#
#        for j in range(4):
#            # time -> time_next:
#            #new_img = p_sample_with_mask(module, params, k1, img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#            aux, k2, k3 = random.split(aux, 3)
#                
#            # Forward noising
#            # Perform markov transition to previous timestep
#            noise = random.normal(k2, img.shape)
#            log_snr = logsnr_schedule_cosine(time)
#            log_snr_next = logsnr_schedule_cosine(time_next)
#            log_snr_padded = right_pad_dims_to(img, log_snr)
#            log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
#            alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
#                jax.nn.sigmoid(-log_snr_padded)
#            )
#            alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
#                jax.nn.sigmoid(-log_snr_next_padded)
#            )
#
#            alpha_t = alpha
#            alpha_s = alpha_next
#            sigma_t = sigma
#            sigma_s = sigma_next
#            zs = new_img
#
#            alpha_ts = alpha_t / alpha_s
#            sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
#            sigma_ts = jnp.sqrt(sigma_ts_sq)
#            new_img_with_noise = zs * alpha_ts + noise * sigma_ts
#
#            # Denoise once more
#            new_img = p_sample_with_mask(module, params, k3, new_img_with_noise, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#        return new_img
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img

#@partial(jit, static_argnums=(0, 4, 5))
def ddpm_sample_with_mask(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return p_sample_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)

        #k1, k2, k3 = random.split(keys[i], 3)
        #new_img = p_sample_with_mask(module, params, k1, img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)

        ## time -> time_next:
        ##new_img = p_sample_with_mask(module, params, k1, img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
        #    
        ## Forward noising
        ## Perform markov transition to previous timestep
        #noise = random.normal(k2, img.shape)
        #log_snr = logsnr_schedule_cosine(time)
        #log_snr_next = logsnr_schedule_cosine(time_next)
        #log_snr_padded = right_pad_dims_to(img, log_snr)
        #log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
        #alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
        #    jax.nn.sigmoid(-log_snr_padded)
        #)
        #alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
        #    jax.nn.sigmoid(-log_snr_next_padded)
        #)

        #alpha_t = alpha
        #alpha_s = alpha_next
        #sigma_t = sigma
        #sigma_s = sigma_next
        #zs = new_img

        #alpha_ts = alpha_t / alpha_s
        #sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
        #new_img_with_noise = zs * alpha_ts + noise * sigma_ts_sq

        ## Denoise once more
        #new_img_resampled = p_sample_with_mask(module, params, k3, new_img_with_noise, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
        #return jax.lax.cond(time_next == 0, lambda x: x[0], lambda x: x[1], (new_img, new_img_resampled))

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def slow_sample(module, params, key, img, num_sample_steps=100, **kwargs):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)
    for i in tqdm(range(num_sample_steps)):
        time = steps[i]
        time_next = steps[i + 1]
        img = p_sample(module, params, keys[i], img, time, time_next, **kwargs)
        print("img:", img.shape)

    img = jnp.clip(img, -1, 1)
    return img

#@partial(jit, static_argnums=(0, 4, 5))
def ddim_sample(
    module, params, key, img, num_sample_steps=100, objective="v", **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return ddim_sample_step(module, params, keys[i], img, time, time_next, objective=objective, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img


def ddim_sample_step(module, params, key, x, time, time_next, objective="v", schedule=logsnr_schedule_cosine,**kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)
    #x_start = dynamic_threshold(x_start)

    model_mean = alpha_next * x_start + (sigma_next / sigma) * (x - alpha * x_start)
    return model_mean

def ddim_sample_step_with_mask(module, params, key, x, time, time_next, objective="v", mask=None, gt=None, schedule=logsnr_schedule_cosine, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    
    gt_noise_key, step_noise_key = random.split(key)

    # Sample noised ground truth for timestep
    gt_noise = random.normal(gt_noise_key, gt.shape)
    noised_gt, _ = q_sample(gt, time, gt_noise, schedule=schedule) 
    x = noised_gt * mask + (1-mask) * x

    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred

    x_start = jnp.clip(x_start, -1, 1)
    model_mean = alpha_next * x_start + (sigma_next / sigma) * (x - alpha * x_start)
    return jax.lax.cond(time_next == 0, lambda m: (mask * gt) + (1-mask) * m, lambda m: m, model_mean)


def ddim_sample_with_mask(module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs):
    steps = jnp.linspace(0.0, 1.0, num_sample_steps + 1)

    t_T = num_sample_steps + 1
    jump_len = 4
    jump_n_sample = 4
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t-1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t=t+1
                ts.append(t)

    print("ts:", ts)
    n = len(ts)-1
    print("n:", n)
    ts = jnp.array(ts)

    print(steps[ts])

    keys = random.split(key, n)

    for i in tqdm(range(n)):
        index = ts[i]
        index_next = ts[i+1]

        time = steps[index]
        time_next = steps[index_next]

        print("index:", index, "next:", index_next)
        print("time:", time, "next:", time_next)

        if time > time_next:
            print("denoising")
            img = ddim_sample_step_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
        else:
            print("noising")
            k2, k3 = random.split(keys[i])

            # Forward noising
            # Perform markov transition to previous timestep
            noise = random.normal(k2, img.shape)
            log_snr = logsnr_schedule_cosine(time)
            log_snr_next = logsnr_schedule_cosine(time_next)
            log_snr_padded = right_pad_dims_to(img, log_snr)
            log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
            alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
                jax.nn.sigmoid(-log_snr_padded)
            )
            alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
                jax.nn.sigmoid(-log_snr_next_padded)
            )

            alpha_t = alpha_next
            alpha_s = alpha
            sigma_t = sigma_next
            sigma_s = sigma
            zs = img

            alpha_ts = alpha_t / alpha_s
            sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
            sigma_ts = jnp.sqrt(sigma_ts_sq)
            img = zs * alpha_ts + noise * sigma_ts
        print("new img:", img.min(), img.max(), img.mean(), img.shape)

    #img = jax.lax.fori_loop(0, n, step, img)
    img = jnp.clip(img, -1, 1)
    return img

#@partial(jit, static_argnums=(0, 4, 5))
#def ddim_sample_with_mask(
#    module, params, key, img, num_sample_steps=100, objective="v", mask=None, gt=None, **kwargs
#):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        return ddim_sample_step_with_mask(module, params, keys[i], img, time, time_next, objective=objective, mask=mask, gt=gt, **kwargs)
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img

def adjusted_ddim_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -jnp.expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)

    xvar = 0.1 / (2 + squared_alpha / squared_sigma)
    eps = (x - alpha * x_start) / sigma
    zs_var = (alpha_next - alpha * sigma_next / sigma) ** 2 * xvar
    
    # Change this for your resolution
    # TOOD: just read out x.shape[1,2]
    #d = 256 * 256
    d = 128 * 128

    model_mean = alpha_next * x_start + jnp.sqrt(squared_sigma_next + (d/jnp.linalg.norm(eps)**2) * zs_var) * eps
    return model_mean


#@partial(jit, static_argnums=(0, 4))
def adjusted_ddim_sample(
    module, params, key, img, num_sample_steps=100, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return adjusted_ddim_sample_step(module, params, keys[i], img, time, time_next, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

#def dpm1_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
#    log_snr = logsnr_schedule_cosine(time)
#    log_snr_next = logsnr_schedule_cosine(time_next)
#
#    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
#        log_snr_next
#    )
#    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
#        -log_snr_next
#    )
#
#    alpha, sigma, alpha_next, sigma_next = map(
#        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
#    )
#
#    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
#    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
#    if objective == "v":
#        eps = sigma * x + alpha * pred
#    elif objective == "eps":
#        eps = pred
#    elif objective == "start":
#        eps = (x - alpha * pred) / sigma
#    h_i = (log_snr_next - log_snr) * 0.5
#    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i) * eps
#
#
#@partial(jit, static_argnums=(0, 4))
#def dpm1_sample(
#    module, params, key, img, num_sample_steps=100, **kwargs
#):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        return dpm1_sample_step(module, params, keys[i], img, time, time_next, **kwargs)
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img
#
#
#def dpm2_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
#    log_snr = logsnr_schedule_cosine(time)
#    log_snr_next = logsnr_schedule_cosine(time_next)
#
#    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
#        log_snr_next
#    )
#    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
#        -log_snr_next
#    )
#
#    alpha, sigma, alpha_next, sigma_next = map(
#        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
#    )
#
#    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
#    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
#    #eps = sigma * x + alpha * v
#
#    x_start = alpha * x - sigma * pred
#    x_start = jnp.clip(x_start, -1, 1)
#    eps = (x - alpha * x_start) / sigma
#
#    #pred = module.apply(params, x, time=batch_log_snr, **kwargs)
#    #if objective == "v":
#    #    x_start = alpha * x - sigma * pred
#    #elif objective == "eps":
#    #    x_start = (x - sigma * pred) / alpha
#    #elif objective == "start":
#    #    x_start = pred
#    #x_start = jnp.clip(x_start, -1, 1)
#    #eps = (x - alpha * x_start) / sigma
#
#    s_i = (log_snr + log_snr_next) * 0.5
#    h_i = (log_snr_next - log_snr) * 0.5
#
#    alpha_si = jnp.sqrt(jax.nn.sigmoid(s_i))
#    sigma_si = jnp.sqrt(jax.nn.sigmoid(-s_i))
#
#    #print("alpha:", alpha_si, sigma_si)
#
#    u_i = (alpha_si / alpha) * x - sigma_si * jnp.expm1(h_i * 0.5) * eps
#    
#    batch_si_logsnr = repeat(s_i, " -> b", b=x.shape[0])
#    pred = module.apply(params, u_i, time=batch_si_logsnr, **kwargs)
#    #eps_si = sigma_si * u_i + alpha_si * v
#
#    x_start = alpha_si * u_i - sigma_si * pred
#    x_start = jnp.clip(x_start, -1, 1)
#    eps_si = (u_i - alpha_si * x_start) / sigma_si
#
#    #if objective == "v":
#    #    x_start = alpha_si * u_i - sigma_si * pred
#    #elif objective == "eps":
#    #    x_start = (u_i - sigma_si * pred) / alpha_si
#    #elif objective == "start":
#    #    x_start = pred
#    #x_start = jnp.clip(x_start, -1, 1)
#    #eps_si = (u_i - alpha_si * x_start) / sigma_si
#    
#    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i) * eps_si
#
#
#@partial(jit, static_argnums=(0, 4))
#def dpm2_sample(
#    module, params, key, img, num_sample_steps=100, **kwargs
#):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        return dpm2_sample_step(module, params, keys[i], img, time, time_next, **kwargs)
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img
#
#
#def dpm3_sample_step(module, params, key, x, time, time_next, objective="v", **kwargs) -> jax.Array:
#    log_snr = logsnr_schedule_cosine(time)
#    log_snr_next = logsnr_schedule_cosine(time_next)
#    c = -jnp.expm1(log_snr - log_snr_next)
#
#    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
#        log_snr_next
#    )
#    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
#        -log_snr_next
#    )
#
#    alpha, sigma, alpha_next, sigma_next = map(
#        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
#    )
#
#    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
#    v = module.apply(params, x, time=batch_log_snr, **kwargs)
#    eps = sigma * x + alpha * v
#
#    r1 = 1./3.
#    r2 = 2./3.
#    h_i = (log_snr_next - log_snr) # in logsnr space
#    
#    s2im1 = log_snr + r1 * h_i
#    s2i = log_snr + r2 * h_i
#    
#    alpha_s2im1 = jnp.sqrt(jax.nn.sigmoid(s2im1))
#    sigma_s2im1 = jnp.sqrt(jax.nn.sigmoid(-s2im1))
#
#    u2im1 = (alpha_s2im1 / alpha) * x - sigma_s2im1 * jnp.expm1(r1 * h_i * 0.5) * eps
#
#    batch_s2im1 = repeat(s2im1, " -> b", b=x.shape[0])
#    v = module.apply(params, u2im1, time=batch_s2im1, **kwargs)
#    eps2im1 = sigma_s2im1 * x + alpha_s2im1 * v
#    d2im1 = eps2im1 - eps
#
#    alpha_s2i = jnp.sqrt(jax.nn.sigmoid(s2i))
#    sigma_s2i = jnp.sqrt(jax.nn.sigmoid(-s2i))
#    u2i = (alpha_s2i / alpha) * x - sigma_s2i * jnp.expm1(r2 * h_i * 0.5) * eps - (sigma_s2i * r2 / r1) * (jnp.expm1(r2 * h_i * 0.5) / (r2 * h_i) - 1) * d2im1
#
#    batch_s2i = repeat(s2i, " -> b", b=x.shape[0])
#    v = module.apply(params, u2i, time=batch_s2i, **kwargs)
#    eps2i = sigma_s2i * x + alpha_s2i * v
#    d2i = eps2i - eps
#    
#    return (alpha_next / alpha) * x - sigma_next * jnp.expm1(h_i * 0.5) * eps - (sigma_next / r2) * (jnp.expm1(h_i * 0.5) / (h_i * 0.5) - 1) * d2i
#
#
#@partial(jit, static_argnums=(0, 4))
#def dpm3_sample(
#    module, params, key, img, num_sample_steps=100, **kwargs
#):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        return dpm3_sample_step(module, params, keys[i], img, time, time_next, **kwargs)
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img
#
#def dpmpp_2s_sample_step(module, params, key, x, time, time_next, objective="v",schedule=logsnr_schedule_cosine **kwargs) -> jax.Array:
#    log_snr = schedule(time)
#    log_snr_next = schedule(time_next)
#    c = -jnp.expm1(log_snr - log_snr_next)
#
#    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
#        log_snr_next
#    )
#    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
#        -log_snr_next
#    )
#
#    alpha, sigma, alpha_next, sigma_next = map(
#        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
#    )
#
#    batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
#
#    pred = module.apply(params, x, time=batch_log_snr, **kwargs)
#    if objective == "v":
#        x_start = alpha * x - sigma * pred
#    elif objective == "eps":
#        x_start = (x - sigma * pred) / alpha
#    elif objective == "start":
#        x_start = pred
#    x_start = jnp.clip(x_start, -1, 1)
#
#    hi = (log_snr_next - log_snr) * 2
#    si = schedule(time + (time_next - time) * 0.5)
#    lam_si = si * 0.5
#    alpha_si, sigma_si = jnp.sqrt(jax.nn.sigmoid(si)), jnp.sqrt(jax.nn.sigmoid(-si))
#
#    ri = (lam_si - log_snr * 0.5) / hi
#    ui = (sigma_si / sigma) * x - alpha_si * jnp.expm1(-ri * hi) * x_start
#    
#    # Get start at ui,si
#    batch_si = repeat(si, " -> b", b=x.shape[0])
#    pred = module.apply(params, ui, time=batch_si, **kwargs)
#    if objective == "v":
#        x_start_ui = alpha_si * ui - sigma_si * pred
#    elif objective == "eps":
#        x_start_ui = (ui - sigma_si * pred) / alpha_si
#    elif objective == "start":
#        x_start_ui = pred
#    x_start_ui = jnp.clip(x_start_ui, -1, 1)
#    di = (1 - 1 / (2 * ri)) * x_start + 1/(2*ri) * x_start_ui
#    return (sigma_next / sigma) * x - alpha_next * jnp.expm1(-hi) * di
#
##@partial(jit, static_argnums=(0, 4, 5))
#def dpmpp_2s_sample(
#    module, params, key, img, num_sample_steps=100, objective="v", **kwargs
#):
#    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
#    keys = random.split(key, num_sample_steps)
#
#    @loop_tqdm(n=num_sample_steps, desc="sampling step")
#    def step(i, img):
#        time = steps[i]
#        time_next = steps[i + 1]
#        return dpmpp_2s_sample_step(module, params, keys[i], img, time, time_next, **kwargs)
#
#    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
#    img = jnp.clip(img, -1, 1)
#    return img


def ddim_sample_step_reverse(module, params, key, x, time, time_next, objective="v", schedule=logsnr_schedule_cosine,**kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)

    squared_alpha, squared_alpha_next = jax.nn.sigmoid(log_snr), jax.nn.sigmoid(
        log_snr_next
    )
    squared_sigma, squared_sigma_next = jax.nn.sigmoid(-log_snr), jax.nn.sigmoid(
        -log_snr_next
    )

    alpha, sigma, alpha_next, sigma_next = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next)
    )

    batch_log_snr = repeat(log_snr_next, " -> b", b=x.shape[0])
    pred = module.apply(params, x, time=batch_log_snr, **kwargs)

    if objective == "v":
        x_start = alpha * x - sigma * pred
    elif objective == "eps":
        x_start = (x - sigma * pred) / alpha
    elif objective == "start":
        x_start = pred
    x_start = jnp.clip(x_start, -1, 1)

    model_mean = alpha_next * x_start + (sigma_next / sigma) * (x - alpha * x_start)

    #n = 20
    #rho = 1.0/n
    #guess = model_mean
    #for i in range(n):
    #    z_prime = ddim_sample_step(module, params, key, guess, time_next, time, objective=objective, schedule=schedule, **kwargs)
    #    guess = guess - rho * (z_prime - x)
    #model_mean = guess

    return model_mean

def ddim_sample_reverse(
    module, params, key, img, num_sample_steps=100, objective="v", **kwargs
):
    steps = jnp.linspace(0.0, 1.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps+1)

    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return ddim_sample_step_reverse(module, params, keys[i], img, time, time_next, objective=objective, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    return img

def custom_repaint(module, params, key, img, num_sample_steps=100, objective="v", schedule=logsnr_schedule_cosine, **kwargs):
    key, noising_key = random.split(key)

    time = 0
    time_next = 0.5

    noise = random.normal(noising_key, img.shape)
    img, _ = q_sample(x_start=img, times=time_next, noise=noise, schedule=schedule)

    ## Forward noising
    ## Perform markov transition to previous timestep
    #noise = random.normal(noising_key, img.shape)
    #log_snr = logsnr_schedule_cosine(time)
    #log_snr_next = logsnr_schedule_cosine(time_next)
    #log_snr_padded = right_pad_dims_to(img, log_snr)
    #log_snr_next_padded = right_pad_dims_to(img, log_snr_next)
    #alpha, sigma = jnp.sqrt(jax.nn.sigmoid(log_snr_padded)), jnp.sqrt(
    #    jax.nn.sigmoid(-log_snr_padded)
    #)
    #alpha_next, sigma_next = jnp.sqrt(jax.nn.sigmoid(log_snr_next_padded)), jnp.sqrt(
    #    jax.nn.sigmoid(-log_snr_next_padded)
    #)
    #alpha_t = alpha_next
    #alpha_s = alpha
    #sigma_t = sigma_next
    #sigma_s = sigma
    #zs = img
    #alpha_ts = alpha_t / alpha_s
    #sigma_ts_sq = (sigma_t * sigma_t) - (alpha_ts * alpha_ts) * (sigma_s * sigma_s)
    #sigma_ts = jnp.sqrt(sigma_ts_sq)
    #img = zs * alpha_ts + noise * sigma_ts
    #img = jnp.clip(img, -1, 1)

    # Denoising
    steps = jnp.linspace(time_next, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)
    @loop_tqdm(n=num_sample_steps, desc="sampling step")
    def step(i, img):
        time = steps[i]
        time_next = steps[i + 1]
        return simple_p_sample(module, params, keys[i], img, time, time_next, objective=objective, schedule=schedule, **kwargs)

    img = jax.lax.fori_loop(0, num_sample_steps, step, img)
    img = jnp.clip(img, -1, 1)
    return img

def dynamic_threshold(x, quantile=0.95):
    s = jnp.clip(jnp.quantile(jnp.abs(rearrange(x, 'b ... -> b (...)')), quantile, axis = -1), min=1)
    s = right_pad_dims_to(x, s)
    return jnp.clip(x, -s, s) / s

def dpmpp_2m_sample_step(module, params, key, z_t, z_a, time, time_next, time_prev, objective="v",schedule=logsnr_schedule_cosine, **kwargs) -> jax.Array:
    log_snr = schedule(time)
    log_snr_next = schedule(time_next)
    log_snr_prev = schedule(time_prev)

    squared_alpha, squared_alpha_next, squared_alpha_prev = map(jax.nn.sigmoid, (log_snr, log_snr_next, log_snr_prev))
    squared_sigma, squared_sigma_next, squared_sigma_prev = map(jax.nn.sigmoid, (-log_snr, -log_snr_next, -log_snr_prev))
    alpha, sigma, alpha_next, sigma_next, alpha_prev, sigma_prev = map(
        jnp.sqrt, (squared_alpha, squared_sigma, squared_alpha_next, squared_sigma_next, squared_alpha_prev, squared_sigma_prev)
    )

    h_t = (log_snr - log_snr_prev) * 0.5
    h_s = (log_snr_next - log_snr) * 0.5
    r_s = h_t / h_s
    
    # Compute x0(z_t)
    batch_log_snr = repeat(log_snr, " -> b", b=z_t.shape[0])
    pred_t = module.apply(params, z_t, time=batch_log_snr, **kwargs)
    if objective == "v":
        x_start = alpha * z_t - sigma * pred_t
    elif objective == "eps":
        x_start = (z_t - sigma * pred_t) / alpha
    elif objective == "start":
        x_start = pred_t
    #x_start = jnp.clip(x_start, -1, 1)
    x_start = dynamic_threshold(x_start)

    # Compute x0(z_a)
    batch_log_snr_prev = repeat(log_snr_prev, " -> b", b=z_a.shape[0])
    pred_a = module.apply(params, z_a, time=batch_log_snr_prev, **kwargs)
    if objective == "v":
        x_start_prev= alpha_prev * z_a - sigma_prev * pred_a
    elif objective == "eps":
        x_start_prev = (z_a - sigma_prev * pred_a) / alpha_prev
    elif objective == "start":
        x_start_prev = pred_a
    #x_start_prev = jnp.clip(x_start_prev, -1, 1)
    x_start_prev = dynamic_threshold(x_start_prev)

    inv_two_r_s = 1.0 / (2 * r_s)
    D_i = (1 + inv_two_r_s) * x_start - inv_two_r_s * x_start_prev
    model_mean = (sigma_next / sigma) * z_t - alpha_next * jnp.expm1(-h_s) * D_i
    return model_mean

def dpmpp_2m_sample(
    module, params, key, img, num_sample_steps=100, objective="v", schedule=logsnr_schedule_cosine, **kwargs
):
    steps = jnp.linspace(1.0, 0.0, num_sample_steps + 1)
    keys = random.split(key, num_sample_steps)

    # First step
    time = steps[0]
    time_next = steps[1]
    z_t = ddim_sample_step(module, params, keys[0], img, time, time_next, objective=objective, schedule=schedule, **kwargs)

    # State is first: previous image, second: current image
    state = (img, z_t)
    
    # Second order steps
    @loop_tqdm(n=num_sample_steps-1, desc="sampling step")
    def step(j, state):
        i = j + 1
        z_a, z_t = state
        time_prev = steps[i-1]
        time = steps[i]
        time_next = steps[i + 1]
        z_s = dpmpp_2m_sample_step(module, params, keys[i], z_t, z_a, time, time_next, time_prev, objective=objective, schedule=schedule, **kwargs)
        return (z_t, z_s)

    _, img = jax.lax.fori_loop(0, num_sample_steps-1, step, state)
    img = jnp.clip(img, -1, 1)
    return img
