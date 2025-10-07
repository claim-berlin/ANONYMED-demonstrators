#!/usr/bin/env python3

# Project Rocinante

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import flax
from jax import jit, random, numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import math
import argparse
import wandb
from glob import glob
import yaml
from typing import NamedTuple
from tqdm import tqdm
from grain.python import DataLoader, ReadOptions, Batch, ShardOptions, IndexSampler
from einops import reduce, repeat
from functools import partial
import cloudpickle
import pickle
import utils
import h5py
from dataset import SliceDS#, SingleSliceDS
from sampling import q_sample, right_pad_dims_to, simple_ddpm_sample, simple_ddpm_sample_with_mask, simple_repaint, logsnr_schedule_cosine, ddim_sample
#from unet import UNet
#from dit import DiT
#from adm import ADM
#from uvit import UViT
from mlp_mixer import MLPMixer
from lr import create_learning_rate_schedule
#from adam_atan2 import adamw_atan2
import dm_pix as pix
#from dit_google import get_b16_model, interpolate_posembed
from dp_accounting import dp_event
from dp_accounting import rdp
from diff_private_opts import dp_rmsprop, dp_adam
#from hungarian_cover import hungarian_cover_tpu_matcher
from jax_tqdm import loop_tqdm

class SingleSliceDS:
    def __init__(self, path, length, rng, crop=False, resample=None):
        self.path = path
        self.rng = rng
        self.crop = crop
        self.resample = resample
        self.length = length
        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        src_slice = self.file["dataset"][index].reshape(32, 32)
        #src_slice = np.clip(src_slice, 0, 100) / 100
        min_v, max_v = src_slice.min(), src_slice.max()
        eps = 1e-8
        src_slice = (src_slice - min_v) / (max_v - min_v + eps)
        src_slice = np.expand_dims(src_slice, -1)
        return src_slice

class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    train_loss: jax.Array
    step: int

weighting_shift_for_resolution = {
    32: 1,
    64: 1,
    128: 1,
    256: -1,
    512: -3,
    1024: -4
}

def main(args):
    print("config:", args)
    dtype = jnp.bfloat16 if args.bfloat16 else jnp.float32
    batch_size = int(args.batch_size)
    print("dtype:", dtype)

    target_resolution = 32#args.resolution
    target_shape = (target_resolution, target_resolution)
    shifted_cosine_schedule = partial(logsnr_schedule_cosine, shift=target_resolution)
    weighting_shift = weighting_shift_for_resolution[target_resolution]
    print("target shape:", target_shape)
    print("weighting shift:", weighting_shift)

    device_count = jax.device_count()
    print("device count:", device_count)
    print("devices:", jax.devices())

    if args.arch == "adm":
        print("default adm")
        module = ADM(dim=128, channels=1, dtype=dtype)
    elif args.arch == "big-adm":
        print("big adm")
        module = ADM(dim=256, channels=1, dtype=dtype)
    elif args.arch == "uvit":
        print("uvit")
        module = UViT(dim=128, channels=1, dtype=dtype)
    elif args.arch == "unet":
        print("unet")
        module = UNet(dim=64, dtype=dtype)
    elif args.arch == "mlp":
        print("mlp")
        # 2, 32
        module = MLPMixer(
            patch_size=2,
            hidden_size=64,
            depth=12,
            in_channels=1,
            dtype=dtype
        )
    elif args.arch == "custom":
        print("custom")
        module = get_model()

    elif args.arch == "dit":
        print("dit")
        num_tokens = (target_resolution // 16) ** 2 + 1
        module, params = get_b16_model("../modality_jax/ViT-B_16.npz", num_tokens=num_tokens)

        #module = DiT(
        #    patch_size=16,
        #    hidden_size=1024,
        #    depth=24,
        #    num_heads=16,
        #    in_channels=1,
        #    dtype=dtype
        #)
        #module = DiT(
        #    patch_size=4,
        #    hidden_size=16,
        #    depth=12,
        #    num_heads=4,
        #    in_channels=1,
        #    dtype=dtype
        #)

    if args.wandb:
        wandb.init(project="expanse_paper", config=args)

    if args.warmup:
        print("warming up lr")
        warmup_steps = args.total_steps * 0.3
        print("number of warmup steps:", warmup_steps)
        lr = create_learning_rate_schedule(total_steps=args.total_steps, base=args.lr, decay_type="cosine", warmup_steps=warmup_steps, linear_end=0)
    else:
        lr = args.lr

    rng = np.random.default_rng(args.seed)

    #if args.opt == "adamw":
    #    print("using default version")
    #    opt = optax.adamw(learning_rate=lr, b1=0.9, b2=0.99, eps=1e-12, weight_decay=0)
    #elif args.opt == "adamw-atan2":
    #    print("using atan2 version")
    #    opt = adamw_atan2(learning_rate=lr, b1=0.9, b2=0.99, weight_decay=0)
    #elif args.opt == "sgd":
    #    print("sgd")
    #    opt = optax.sgd(learning_rate=lr, momentum=0.9, nesterov=True)
    #elif args.opt == "lion":
    #    print("lion")
    #    opt = optax.lion(learning_rate=lr, weight_decay=0)
    #elif args.opt == "dpsgd":
    print("diff private sgd")
    L2_NORM_CLIP = 1.0
    NOISE_MULTIPLIER = 1.3
    TARGET_DELTA = 1e-6
    #opt = optax.contrib.dpsgd(learning_rate=lr, l2_norm_clip=L2_NORM_CLIP, noise_multiplier=NOISE_MULTIPLIER, seed=args.seed, momentum=.9, nesterov=True)
    #opt = dp_rmsprop(learning_rate=lr, l2_norm_clip=L2_NORM_CLIP, noise_multiplier=NOISE_MULTIPLIER, seed=args.seed, momentum=0.9, nesterov=True, bias_correction=True)
    opt = dp_adam(learning_rate=lr, l2_norm_clip=L2_NORM_CLIP, noise_multiplier=NOISE_MULTIPLIER, seed=args.seed, b1=0.9, b2=0.99, eps=1e-12)

    #if args.clip:
    #    print("use grad clip")
    #    opt = optax.chain(optax.clip_by_global_norm(1.0), opt)

    def midpoint(f, y, num_steps, h=0.1):
        times = jnp.linspace(0., 1., num=num_steps)
        times = repeat(times, "d -> d b", b=y.shape[0])

        @loop_tqdm(n=num_steps, desc="sampling step")
        def step(i, y):
            t = times[i]
            ykhalf = y + (h/2) * f(t, y)
            return y + h * f(t + h/2, ykhalf)
        return jax.lax.fori_loop(0, num_steps, step, y)

    def euler(f, y, num_steps, h=0.1):
        times = jnp.linspace(0., 1., num=num_steps)
        times = repeat(times, "d -> d b", b=y.shape[0])

        @loop_tqdm(n=num_steps, desc="sampling step")
        def step(i, y):
            t = times[i]
            return y + h * f(t, y)
        return jax.lax.fori_loop(0, num_steps, step, y)

    @partial(jit, static_argnums=(0,))
    def flow_sample(module, params, noise):
        num_steps = 4
        step_size = 1.0 / num_steps
        def ode_fn(t,y):
            return module.apply(params, y, time=t)
        return midpoint(ode_fn, noise, num_steps=num_steps, h=step_size)

    def flow_loss(params, batch, key):
        x = batch
        timekey, noisekey = random.split(key)

        data = x * 2 - 1
        #noise = random.normal(noisekey, x.shape)
        noise = random.beta(noisekey, 3/2, 3/2, shape=x.shape)
        t = random.uniform(timekey, (x.shape[0],))

        # Predict flow
        t_padded = right_pad_dims_to(data, t)
        noised = t_padded * data + (1-t_padded) * noise
        flow = data - noise
        predicted_flow = module.apply(params, noised, time=t, train=True)
        loss = jnp.mean(jnp.square(predicted_flow - flow))
        return loss

    def loss(params, batch, key):
        timekey, noisekey, dropoutkey = random.split(key, 3)
        x = batch
        x = x * 2 - 1

        # VDM low-discrepancy sampler from https://arxiv.org/pdf/2107.00630
        t0 = random.uniform(timekey)
        n_batch = x.shape[0]
        times = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    
        x_start = x
        noise = random.normal(noisekey, x_start.shape)
        #noise = draw_noise(noisekey, x_start)

        x_noised, log_snr = q_sample(x_start=x_start, times=times, noise=noise, schedule=shifted_cosine_schedule)
        model_out = module.apply(params, x_noised, time=log_snr, train=True, rngs={"dropout": dropoutkey})
        target = noise

        loss = jnp.square(model_out - target)
        loss = reduce(loss, "b ... -> b", "mean")

        # Use sigmoid weighting from Kingma et al
        weight = jax.nn.sigmoid(weighting_shift - log_snr)

        loss = (loss * weight).mean()
        return loss

    @partial(jax.pmap, axis_name="device", in_axes=(0,0,0))
    def update(state, batch, key):
        grad_fn = jax.value_and_grad(flow_loss)
        #if args.opt == "dpsgd":
        batch = jax.tree.map(lambda x: x[:, None], batch)
        # Uses jax.vmap across the batch to extract per-example gradients.
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, None))

        train_loss_value, grads = grad_fn(state.params, batch, key)

        grads = jax.lax.pmean(grads, "device")
        updates, opt_state = opt.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(
            params=params,
            opt_state=opt_state,
            train_loss=jnp.mean(train_loss_value),
            step=state.step+1
        )


    train_slices_path = "../data/atlas_slices_32x.h5"
    val_slices_path = train_slices_path

    with h5py.File(train_slices_path, "r") as f:
        train_len = f["dataset"].shape[0]
    with h5py.File(val_slices_path, "r") as f:
        val_len = f["dataset"].shape[0]

    train_ds = SingleSliceDS(train_slices_path, length=train_len, rng=rng, resample=target_shape)
    val_ds = SingleSliceDS(val_slices_path, length=val_len, rng=rng, resample=target_shape)

    print("train ds:", train_len)
    print("num batches:", math.ceil(train_len / batch_size))

    #test_slices_path = "../hemorrhage_data/slices_train_with_seg_new.h5"
    #with h5py.File(test_slices_path, "r") as f:
    #    test_len = f["dataset"].shape[0]
    #test_ds = SliceDS(test_slices_path, length=test_len, rng=rng, resample=target_shape)

    # Setup dataloaders
    shard_opts = ShardOptions(0, 1)
    read_opts = ReadOptions(num_threads=16, prefetch_buffer_size=1)

    def train_cycle():
        while True:
            seed = int(rng.random() * (2**32 - 1))
            train_sampler = IndexSampler(
                train_len, shard_opts, shuffle=True, seed=seed
            )
            train_dl = DataLoader(
                data_source=train_ds,
                sampler=train_sampler,
                worker_count=4,
                shard_options=shard_opts,
                read_options=read_opts,
                operations=[Batch(batch_size=batch_size, drop_remainder=False)],
            )
            it = iter(train_dl)
            for x in it:
                yield x

    #if args.opt == "dpsgd":
    num_examples = train_len
    if num_examples * TARGET_DELTA > 1:
        print("delta too high")
    else:
        print("you choose delta wisely. Hooray!")

    def compute_epsilon(steps, num_examples, target_delta):
        q = batch_size / num_examples
        orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
        accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
        accountant.compose(
        dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(NOISE_MULTIPLIER)), steps)
        return accountant.get_epsilon(target_delta)

    val_sampler = IndexSampler(len(val_ds), shard_opts, shuffle=True, seed=args.seed)
    val_dl = DataLoader(
        data_source=val_ds,
        sampler=val_sampler,
        worker_count=4,
        shard_options=shard_opts,
        read_options=read_opts,
        operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    )
    #test_sampler = IndexSampler(len(test_ds), shard_opts, shuffle=True, seed=args.seed)
    #test_dl = DataLoader(
    #    data_source=test_ds,
    #    sampler=test_sampler,
    #    worker_count=4,
    #    shard_options=shard_opts,
    #    read_options=read_opts,
    #    operations=[Batch(batch_size=batch_size, drop_remainder=False)],
    #)

    key = random.key(args.seed)
    key, initkey = random.split(key)

    # Setup state
    if args.load is not None:
        with open(args.load, "rb") as f:
            state = pickle.load(f)
        print("train loss:", state.train_loss)
        print("steps:", state.step)

        if args.arch == "dit":

            # Check resolution
            num_tokens = (target_resolution // 16) ** 2 + 1
            params = state.params
            posemb = params["params"]["Transformer"]["posembed_input"]["pos_embedding"]
            print("found:", posemb.shape[1], "expected:", num_tokens)

            if posemb.shape[1] != num_tokens:
                print("resizing to resolution")
                posemb = interpolate_posembed(posemb, num_tokens, has_class_token=True)
                params["params"]["Transformer"]["posembed_input"]["pos_embedding"] = posemb
                initial_opt_state = opt.init(params)
                state = TrainingState(
                    params=params,
                    opt_state=initial_opt_state,
                    train_loss=state.train_loss,
                    step=state.step,
                )
    else:
        if args.arch == "dit":
            initial_opt_state = opt.init(params)
            state = TrainingState(
                params=params,
                opt_state=initial_opt_state,
                train_loss=None,
                step=0,
            )
        else:
            train_dl = train_cycle()
            x = next(iter(train_dl))
            #dummy_time = jnp.array([0.0])
            dummy_time = jnp.zeros((x.shape[0],))
            initial_params = module.init(initkey, x, time=dummy_time, train=False)
            initial_opt_state = opt.init(initial_params)
            state = TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                train_loss=None,
                step=0,
            )

    #step = int(state.step)
    #print("step:", step)
    #eps = compute_epsilon(steps=step, num_examples=train_len, target_delta=TARGET_DELTA)
    #print("eps:", eps)
    #import sys
    #sys.exit(0)

    if args.train:
        state = flax.jax_utils.replicate(state)

        #train_dl = utils.cycle(train_dl)
        train_dl = train_cycle()
        for step in (p := tqdm(range(args.total_steps))):
            batch = next(train_dl)
            key, updatekey = random.split(key)

            b,h,w,c = batch.shape
            batch = batch.reshape(device_count, -1, h, w, c)
            updatekeys = random.split(updatekey, device_count)

            state = update(state, batch, updatekeys)

            train_loss = flax.jax_utils.unreplicate(state.train_loss)
            p.set_description(f"train_loss: {train_loss}")

            if args.wandb and (
                step % args.log_every_n_steps == args.log_every_n_steps - 1
            ):
                wandb.log({
                    "train_loss": train_loss}, step=step)

            if step % args.validate_every_n_steps == args.validate_every_n_steps - 1:
                if args.save is not None:
                    state = flax.jax_utils.unreplicate(state)
                    with open(args.save, "wb") as f:
                        cloudpickle.dump(state, f)
                    state = flax.jax_utils.replicate(state)

                print("computing epsilon")
                steps = int(flax.jax_utils.unreplicate(state.step))
                eps = compute_epsilon(steps=steps, num_examples=train_len, target_delta=TARGET_DELTA)

                if args.wandb:
                    wandb.log({"eps": eps}, step=step)
                else:
                    print("for target delta", TARGET_DELTA, "epsilon is:", eps)
    
    elif args.sample:
        x = next(iter(train_cycle()))

        batch_size = args.batch_size
        h, w = target_shape
        key, initkey, samplekey = random.split(key, 3)
        
        #img = random.beta(initkey, 3/2, 3/2, shape=(batch_size, h, w, 1))
        #samples = flow_sample(module, state.params, img)

        img = random.normal(initkey, (batch_size, h, w, 1))
        samples = simple_ddpm_sample(
            module=module,
            params=state.params,
            key=samplekey,
            img=img,
            num_sample_steps=args.num_sample_steps,
            objective="eps",
            train=False,
            schedule=shifted_cosine_schedule,
        )
           
        samples = jnp.clip((samples + 1) * 0.5, 0.0, 1.0)

        samples = jnp.concatenate((x, samples), axis=0)
        samples = samples.reshape(-1, h, w)
        img = utils.make_grid(samples, nrow=2, ncol=batch_size)
        utils.save_image(img, "out.png")

    elif args.inpaint:
        #train_dl = train_cycle()
        x, seg = jnp.array(next(iter(test_dl)))
        x = x * 2 - 1

        s = seg.reshape(x.shape[0], -1)
        print(s.min(1), s.max(1), s.sum(1))

        box_w = 32
        box_h = 32
        #h,w = 256, 256
        h,w = target_shape

        b = x.shape[0]

        mask = 1-seg

        #mask = jnp.ones((b, h, w, 1))

        #for i in range(b):
        #    for j in range(2):
        #        rx = rng.integers(0, h-box_w)
        #        ry = rng.integers(0, w-box_h)
        #        mask = mask.at[i, rx:rx+box_w, ry:ry+box_h, :].set(0)
        gt = x
        print("mask", mask.shape, mask.min(), mask.max())
        print("gt:", gt.shape)

        batch_size = x.shape[0]
        key, initkey, samplekey = random.split(key, 3)
        img = random.normal(initkey, (batch_size, h, w, 1))

        x_hat = simple_ddpm_sample_with_mask(
        #x_hat = simple_repaint(
            module=module,
            params=state.params,
            key=samplekey,
            img=img,
            gt=gt,
            mask=mask,
            num_sample_steps=args.num_sample_steps,
            objective="eps",
            train=False,
            schedule=shifted_cosine_schedule
        )

        x = (x + 1) * 0.5
        x_hat = (x_hat + 1) * 0.5
        x_cutout = x * mask
            
        samples = jnp.concatenate((x, x_cutout, x_hat), axis=0)
        samples = jnp.clip(samples, 0.0, 1.0)

        samples = samples.reshape(-1, h, w)
        img = utils.make_grid(samples, nrow=3, ncol=batch_size)
        utils.save_image(img, "out.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--load", type=str, help="path to load pretrained weights from")
    p.add_argument(
        "--save", type=str, help="path to save weight to", default="checkpoint.pkl"
    )
    p.add_argument("--train", action="store_true", help="train the model")
    p.add_argument("--sample", action="store_true", help="sample the model")
    p.add_argument("--reverse", action="store_true", help="reverse sample the model")
    p.add_argument("--eval", action="store_true", help="evaluate the model")
    p.add_argument("--inpaint", action="store_true", help="inpaint using the model")
    p.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=1000,
        help="validate the model every n steps",
    )
    p.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="log the metrics every n steps",
    )
    p.add_argument("--bfloat16", action="store_true", help="use bfloat16 precision")
    p.add_argument("--batch_size", type=int, default=16, help="batch size")
    p.add_argument(
        "--total_steps",
        type=int,
        default=150000,
        help="total number of steps to train for",
    )
    p.add_argument("--seed", type=int, default=42, help="global seed")
    p.add_argument("--num_sample_steps", type=int, default=100, help="sample steps")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--opt", type=str, default="adamw", help="optimizer")
    p.add_argument("--warmup", action="store_true", help="use lr warmup")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--resolution", type=int, default=128, help="image resolution")
    p.add_argument("--arch", type=str, default="adm", help="model architecture")
    p.add_argument("--clip", action="store_true", help="use gradient clipping")
    main(p.parse_args())
