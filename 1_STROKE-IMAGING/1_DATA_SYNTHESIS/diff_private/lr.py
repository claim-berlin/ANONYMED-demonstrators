import jax.numpy as jnp
def create_learning_rate_schedule(total_steps,
                                  base,
                                  decay_type,
                                  warmup_steps,
                                  linear_end=1e-5):
  """Creates learning rate schedule.

  Currently only warmup + {linear,cosine} but will be a proper mini-language
  like preprocessing one in the future.

  Args:
    total_steps: The total number of steps to run.
    base: The starting learning-rate (without warmup).
    decay_type: 'linear' or 'cosine'.
    warmup_steps: how many steps to warm up for.
    linear_end: Minimum learning rate.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type == 'linear':
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    else:
      raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn
