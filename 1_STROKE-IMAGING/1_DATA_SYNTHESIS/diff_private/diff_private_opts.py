from typing import Any, NamedTuple, Optional
import warnings
import jax
from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import transform
from optax._src import linear_algebra
#from optax.contrib._privacy import differentially_private_aggregate
from optax._src import utils
import jax.numpy as jnp
import chex

def per_example_global_norm_clip(
    grads: chex.ArrayTree, l2_norm_clip: float
) -> tuple[chex.ArrayTree, jax.Array]:

  global_grad_norms = jax.vmap(linear_algebra.global_norm)(grads)
  multipliers = jnp.nan_to_num(
      jnp.minimum(l2_norm_clip / global_grad_norms, 1.0), nan=1.0
  )
  num_clipped = jnp.sum(multipliers < 1.0)
  clipped_sum = jax.tree.map(
      lambda g: jnp.tensordot(multipliers, g, axes=1), grads
  )
  return clipped_sum, num_clipped

#Differentially Private Per-
#Sample Adaptive Clipping  DP-PSAC

# https://arxiv.org/pdf/2212.00328
# https://arxiv.org/pdf/2411.03059v1
def adaptive_scaling_clipping(
        grads: chex.ArrayTree, l2_norm_clip: float, r: float = 0.01, s: float = 1
) -> tuple[chex.ArrayTree, jax.Array]:

  global_grad_norms = jax.vmap(linear_algebra.global_norm)(grads)
  #multipliers = jnp.nan_to_num(
  #    jnp.minimum(l2_norm_clip / global_grad_norms, 1.0), nan=1.0
  #)
  multipliers = l2_norm_clip / (s * global_grad_norms + r/ (global_grad_norms + r))

  clipped_sum = jax.tree.map(
      lambda g: jnp.tensordot(multipliers, g, axes=1), grads
  )
  return clipped_sum

class DifferentiallyPrivateAggregateState(NamedTuple):
  """State containing PRNGKey for `differentially_private_aggregate`."""

  rng_key: jax.Array

def canonicalize_key(key_or_seed: jax.Array) -> jax.Array:
  """Canonicalize a random key or an int representing a seed to a random key."""
  if (isinstance(key_or_seed, jax.Array) and jnp.issubdtype(
      key_or_seed.dtype, jax.dtypes.prng_key
  )):
    return key_or_seed
  return jax.random.key(key_or_seed)


def differentially_private_aggregate(
    l2_norm_clip: Optional[float],
    noise_multiplier: float,
    key: jax.Array = None,
    *,
    seed: int = None,  # deprecated
) -> base.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    key: random generator key for noise generation.
    seed: deprecated, use key instead.

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    Unlike other transforms, `differentially_private_aggregate` expects
    the input updates to have a batch dimension in the 0th axis. That is, this
    function expects per-example gradients as input (which are easy to obtain in
    JAX using `jax.vmap`). It can still be composed with other transformations
    as long as it is the first in the chain.

  .. warning::
    Generic gradient aggregation tools like :class:`optax.MultiSteps` or
    :func:`optax.apply_every` won't work correctly with this transformation
    since the whole point of this transformation is to aggregate gradients in a
    specific way.
  """

  if seed is not None:
    warnings.warn(
        '"seed" is deprecated and will be removed in optax 0.2.7, use "key".',
        DeprecationWarning,
    )
    if key is not None:
      raise ValueError('Only one of seed or key can be specified.')
    key = jax.random.key(seed)
  if key is None:
    warnings.warn('Specifying a key will be required in optax 0.2.7.')
    key = jax.random.key(0)
  key = canonicalize_key(key)

  if l2_norm_clip is not None:
    noise_std = l2_norm_clip * noise_multiplier
  else:
    noise_std = noise_multiplier

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=key)

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree.flatten(updates)
    bsize = grads_flat[0].shape[0]

    if l2_norm_clip is not None:
      clipped, _ = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)
    else:
      clipped = grads_flat

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [
        (g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
        for g, r in zip(clipped, rngs)
    ]
    return (
        jax.tree.unflatten(grads_treedef, noised),
        DifferentiallyPrivateAggregateState(rng_key=new_key),
    )

  return base.GradientTransformation(init_fn, update_fn)

def differentially_private_aggregate_adaptive(
    l2_norm_clip: Optional[float],
    noise_multiplier: float,
    key: jax.Array = None,
    *,
    seed: int = None,  # deprecated
) -> base.GradientTransformation:
  """Aggregates gradients based on the DPSGD algorithm.

  Args:
    l2_norm_clip: maximum L2 norm of the per-example gradients.
    noise_multiplier: ratio of standard deviation to the clipping norm.
    key: random generator key for noise generation.
    seed: deprecated, use key instead.

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Abadi et al, 2016 `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. warning::
    Unlike other transforms, `differentially_private_aggregate` expects
    the input updates to have a batch dimension in the 0th axis. That is, this
    function expects per-example gradients as input (which are easy to obtain in
    JAX using `jax.vmap`). It can still be composed with other transformations
    as long as it is the first in the chain.

  .. warning::
    Generic gradient aggregation tools like :class:`optax.MultiSteps` or
    :func:`optax.apply_every` won't work correctly with this transformation
    since the whole point of this transformation is to aggregate gradients in a
    specific way.
  """

  if seed is not None:
    warnings.warn(
        '"seed" is deprecated and will be removed in optax 0.2.7, use "key".',
        DeprecationWarning,
    )
    if key is not None:
      raise ValueError('Only one of seed or key can be specified.')
    key = jax.random.key(seed)
  if key is None:
    warnings.warn('Specifying a key will be required in optax 0.2.7.')
    key = jax.random.key(0)
  key = canonicalize_key(key)

  scaling_coeff = 1.0
  r = 0.1
  noise_std = l2_norm_clip * noise_multiplier / scaling_coeff

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=key)

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree.flatten(updates)
    bsize = grads_flat[0].shape[0]

    #if l2_norm_clip is not None:
    #  clipped, _ = clipping.per_example_global_norm_clip(grads_flat, l2_norm_clip)
    #else:
    #  clipped = grads_flat
    #clipped = adaptive_scaling_clipping(grads_flat, l2_norm_clip, r=r, s=scaling_coeff)
    clipped = [g[0,...] for g in grads_flat]
    #print("clipped:", len(clipped), clipped[0].shape)

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [
        (g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
        for g, r in zip(clipped, rngs)
    ]
    return (
        jax.tree.unflatten(grads_treedef, noised),
        DifferentiallyPrivateAggregateState(rng_key=new_key),
    )

  return base.GradientTransformation(init_fn, update_fn)

def dp_rmsprop(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    centered: bool = False,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    bias_correction: bool = False,
) -> base.GradientTransformation:
  r"""A flexible RMSProp optimizer.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimizer that can be used to switch between several of these variants.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    decay: Decay used to track the magnitude of previous gradients.
    eps: A small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: Initial value of accumulators tracking the magnitude of
      previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
      from a paper, verify the value used by the authors.
    eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
      outside the square root.
    centered: Whether the second moment or the variance of the past gradients is
      used to rescale the latest gradients.
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    bias_correction: Whether to apply bias correction to the estimates of the
      second moments (and first moment if ``centered=True``).

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.rmsprop(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Hinton, `Overview of mini-batch gradient descent`
    <www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_, 2012

    Graves, `Generating Sequences With Recurrent Neural Networks
    <https://arxiv.org/pdf/1308.0850v5>`_, 2014

    Ziyin, `LaProp: Separating Momentum and Adaptivity in Adam`
    <https://arxiv.org/pdf/2002.04839>`_, 2021

  .. warning::
    Default behavior of optax's RMSprop (``eps_in_sqrt=True``) differs from
    Pytorch's implementation and could impact performance.
    If ``eps_in_sqrt=True``, in the denominator, optax uses
    :math:`\sqrt{v + \epsilon}` in the denominator whereas PyTorch uses
    :math:`\sqrt{v} + \epsilon`.
    Using ``eps_in_sqrt=False`` in optax will match PyTorch's behavior.
    See
    https://github.com/google-deepmind/optax/issues/532 for more detail.
  """
  if centered:
    return combine.chain(
        differentially_private_aggregate(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            seed=seed,
        ),
        transform.scale_by_stddev(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            bias_correction=bias_correction,
        ),
        transform.scale_by_learning_rate(learning_rate),
        (
            transform.trace(decay=momentum, nesterov=nesterov)
            if momentum is not None
            else base.identity()
        ),
    )
  return combine.chain(
      differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed,
      ),
      transform.scale_by_rms(
          decay=decay,
          eps=eps,
          initial_scale=initial_scale,
          eps_in_sqrt=eps_in_sqrt,
          bias_correction=bias_correction,
      ),
      transform.scale_by_learning_rate(learning_rate),
      (
          transform.trace(decay=momentum, nesterov=nesterov)
          if momentum is not None
          else base.identity()
      ),
  )

def dp_adam(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""The Adam optimizer.

  Adam is an SGD variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The solver with
      nesterov=True is equivalent to the :func:`optax.nadam` optimizer, and
      described in [Dozat 2016].

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adam(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kingma et al, `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_, 2014

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. warning::
    PyTorch and optax's implementation follow Algorithm 1 of [Kingma et al.
    2014]. Note that TensorFlow used instead the formulation just before Section
    2.1 of the paper. See https://github.com/deepmind/optax/issues/571 for more
    detail.

  .. seealso:: :func:`optax.nadam`, :func:`optax.adamw`.
  """
  return combine.chain(
      differentially_private_aggregate_adaptive(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed,
      ),
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )

def dp_lion(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
  r"""The Lion optimizer.

  Lion is discovered by symbolic program search. Unlike most adaptive optimizers
  such as AdamW, Lion only tracks momentum, making it more memory-efficient.
  The update of Lion is produced through the sign operation, resulting in a
  larger norm compared to updates produced by other optimizers such as SGD and
  AdamW. A suitable learning rate for Lion is typically 3-10x smaller than that
  for AdamW, the weight decay for Lion should be in turn 3-10x larger than that
  for AdamW to maintain a similar strength (lr * wd).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  represent the arguments ``b1`` and ``b2`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function. Let :math:`\lambda` be the weight decay and
  :math:`\theta_t` the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0) = (0)`, representing the intial estimate for the
  first moment. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t`
  and the parameters :math:`\theta_t` and computes updates :math:`u_t` and
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      c_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      u_t &\leftarrow -\alpha_t \cdot \left( sign \left( c_t \right) +
      \lambda \theta_{t} \right)\\
      m_t &\leftarrow \beta_2 \cdot m_{t-1} + (1-\beta_2) \cdot g_t \\
      S_t &\leftarrow (m_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Rate to combine the momentum and the current gradient.
    b2: Exponential decay rate to track the momentum of past gradients.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.

  Returns:
    The corresponding :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lion(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Chen et al, `Symbolic Discovery of Optimization Algorithms
    <https://arxiv.org/abs/2302.06675>`_, 2023
  """
  return combine.chain(
      differentially_private_aggregate(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          seed=seed,
      ),
      transform.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
      transform.scale_by_learning_rate(learning_rate),
  )
