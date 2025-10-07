import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Union, Any, Optional
from einops import rearrange, repeat
import math
from functools import partial
import scipy

def pair(x: Union[int, tuple[int, int]]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)

def timestep_embedding(t: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            (embedding, jnp.zeros_like(embedding[:, :1])), axis=-1
        )
    return embedding

def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1 + scale[:, jnp.newaxis]) + shift[:, jnp.newaxis]

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Any = jnp.float32
  
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mlp_dim, dtype=self.dtype)(x)
        x = nn.silu(x)
        return nn.Dense(x.shape[-1], dtype=self.dtype)(x)

class MixerBlock(nn.Module):
    """Mixer block layer."""
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dtype: Any = jnp.float32
  
    @nn.compact
    def __call__(self, x, c):
        adaln = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    6 * self.channels_mlp_dim,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype
                ),
            ]
        )(c)
        shift_block1, scale_block1, gate_block1, shift_block2, scale_block2, gate_block2 = jnp.split(
            adaln, 6, axis=-1
        )

        y = modulate(nn.LayerNorm()(x), shift_block1, scale_block1)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, dtype=self.dtype,name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y * gate_block1[:, jnp.newaxis]
        y = modulate(nn.LayerNorm()(x), shift_block2, scale_block2)
        return x + MlpBlock(self.channels_mlp_dim, dtype=self.dtype, name='channel_mixing')(y) * gate_block2[:, jnp.newaxis]

class FinalLayer(nn.Module):
    hidden_size: int
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        h, w = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        adaln = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    2 * self.hidden_size,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype
                ),
            ]
        )(c)
        shift, scale = jnp.split(adaln, 2, axis=-1)

        x = modulate(nn.LayerNorm()(x), shift, scale)
        x = nn.Dense(
            features=self.patch_size[0] * self.patch_size[1] * self.channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x)

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=h,
            w=w,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return x

class MLPMixer(nn.Module):
    patch_size: Union[int, tuple[int, int]] = 16
    in_channels: int = 1
    hidden_size: int = 1024
    depth: int = 16
    frequency_embedding_size: int = 256
    out_channels: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: jax.Array, time: jax.Array, condition: Optional[jax.Array] = None, *, train=False
    ) -> jax.Array:
        patch_size = pair(self.patch_size)
        image_size = (x.shape[1], x.shape[2])

        if condition is not None:
            x = jnp.concatenate((x, condition), axis=-1)

        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            dtype=self.dtype,
            name="embedding",
        )(x)

        b,h,w,d = x.shape
        x = rearrange(x, "b h w d -> b (h w) d")

        # Embed time
        t_freq = timestep_embedding(time, self.frequency_embedding_size)
        t_emb = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                    dtype=self.dtype
                ),
                nn.silu,
                nn.Dense(
                    self.hidden_size,
                    kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                    dtype=self.dtype
                ),
            ]
        )(t_freq)

        channels_mlp_dim = self.hidden_size
        tokens_mlp_dim = h * w

        # Transformer blocks
        for _ in range(self.depth):
            x = MixerBlock(tokens_mlp_dim, channels_mlp_dim, dtype=self.dtype)(x, t_emb)

        # Unpatching
        x = FinalLayer(self.hidden_size, image_size, patch_size, self.out_channels, dtype=self.dtype)(
            x, t_emb
        )
        return x
