import copy
import functools
import gc
import io
import logging
import os,sys
import time
from typing import Any
import datasets
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import losses
import numpy as np
import sampling
from diffusions.ddpm import DDPM
import tensorflow as tf
import tensorflow_gan as tfgan
from absl import flags
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
# Keep the import below for registering all model definitions
from models import ddpm, ncsnpp, ncsnv2
from models import utils as mutils
from PIL import Image
import math
import evaluation

FLAGS = flags.FLAGS


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and save it into an image file.

    Pixel values are assumed to be within [0, 1].

    Args:
        ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
        fp: A filename(string) or file object.
        nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        format(Optional):  If omitted, the format to use is determined from the
        filename extension. If a file object was used instead of a filename, this
        parameter should always be used.
    """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and
            all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError("array_like of tensors expected, got {}".format(
        type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                        padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    im = Image.fromarray(np.asarray(ndarr).copy())
    im.save(fp, format=format)


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    print("="*120)
    print("The hyperparameter in config are:", config)
    print("="*120)

    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    rng = jax.random.PRNGKey(config.seed)
    tb_dir = os.path.join(workdir, "tensorboard")
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
    optimizer = train_state.TrainState.create(
        tx=losses.get_optimizer(config),
        params=jax.tree_util.tree_map(lambda x:jnp.array(x),initial_params),
        apply_fn=score_model.apply
    )
    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=rng)  # pytype: disable=wrong-keyword-args

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(checkpoint_meta_dir):
        os.makedirs(checkpoint_meta_dir)

    state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
    initial_step = int(state.step)
    rng = state.rng

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization)

    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = lambda x: x * 2. - 1.
    inverse_scaler = lambda x: (x + 1.) / 2.

    ddpm = DDPM(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    train_step_fn = losses.get_step_fn(ddpm, score_model, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean)

    p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
    eval_step_fn = losses.get_step_fn(ddpm, score_model, train=False, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean)
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                          config.data.image_size, config.data.num_channels)
        sampling_fn = sampling.get_sampling_fn(config, ddpm, score_model, sampling_shape, inverse_scaler, sampling_eps)
    pstate = flax_utils.replicate(state)
    if jax.host_id() == 0:
        logging.info("Starting training loop at step %d." % (initial_step,))
    rng = jax.random.fold_in(rng, jax.host_id())


    # JIT multiple training steps together for faster training
    n_jitted_steps = config.training.n_jitted_steps
    # Must be divisible by the number of steps jitted together
    assert config.training.log_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
           config.training.eval_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"
    
    print("initial step: %d" % (initial_step + 1))
    num_train_step = config.training.n_iters
    print("num train step: %d" % num_train_step)

    for step in range(initial_step, num_train_step + 1, config.training.n_jitted_steps):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        # Execute one training step
        (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
        loss = flax.jax_utils.unreplicate(ploss).mean()
        # logging.info("step: %d" % (step))
        if jax.host_id() == 0 and step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))

        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
            saved_state = flax_utils.unreplicate(pstate)
            saved_state = saved_state.replace(rng=rng)
            # if int(step // config.training.snapshot_freq_for_preemption) != 5:
            checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                        step=step // config.training.snapshot_freq_for_preemption,
                                        keep=1)
            
        if step % config.training.eval_freq == 0:
            eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
            rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            next_rng = jnp.asarray(next_rng)
            (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
            eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
            if jax.host_id() == 0:
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_step:
            # Save the checkpoint.
            if jax.host_id() == 0:
                saved_state = flax_utils.unreplicate(pstate)
                saved_state = saved_state.replace(rng=rng)
                checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                            step=step // config.training.snapshot_freq,
                                            keep=np.inf)

            # Generate and save samples
            if config.training.snapshot_sampling:
                rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                sample_rng = jnp.asarray(sample_rng)
                sample, n = sampling_fn(sample_rng, pstate)
                this_sample_dir = os.path.join(
                    sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
                tf.io.gfile.makedirs(this_sample_dir)
                image_grid = sample.reshape((-1, *sample.shape[2:]))
                nrow = int(np.sqrt(image_grid.shape[0]))
                sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout, nrow=nrow, padding=2)



def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    rng = jax.random.PRNGKey(config.seed + 1)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=1,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = lambda x: x * 2. - 1.
    inverse_scaler = lambda x: (x + 1.) / 2.

    # Initialize model
    rng, model_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
    optimizer = train_state.TrainState.create(
        tx=losses.get_optimizer(config),
        params=initial_params,
        apply_fn=score_model.apply
    )
    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=rng)  # pytype: disable=wrong-keyword-args

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup DDPMs
    ddpm = DDPM(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(ddpm, score_model,
                                       train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean)
        # Pmap (and jit-compile) multiple evaluation steps together for faster execution
        p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step), axis_name='batch', donate_argnums=1)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        additional_dim=None,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")


    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                          config.data.image_size, config.data.image_size,
                          config.data.num_channels)
        sampling_fn = sampling.get_sampling_fn(config, ddpm, score_model, sampling_shape, inverse_scaler, sampling_eps)

    # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
    rng = jax.random.fold_in(rng, jax.host_id())

    # A data class for storing intermediate results to resume evaluation after pre-emption
    @flax.struct.dataclass
    class EvalMeta:
        ckpt_id: int
        sampling_round_id: int
        bpd_round_id: int
        rng: Any

    # Add one additional round to get the exact number of samples as required.
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    num_bpd_rounds = len(ds_bpd) * bpd_num_repeats

    # Restore evaluation after pre-emption
    eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, sampling_round_id=-1, bpd_round_id=-1, rng=rng)
    eval_meta = checkpoints.restore_checkpoint(
        eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

    if eval_meta.bpd_round_id < num_bpd_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = eval_meta.bpd_round_id + 1
        begin_sampling_round = 0

    elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = num_bpd_rounds
        begin_sampling_round = eval_meta.sampling_round_id + 1

    else:
        begin_ckpt = eval_meta.ckpt_id + 1
        begin_bpd_round = 0
        begin_sampling_round = 0

    rng = eval_meta.rng

    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    logging.info("begin checkpoint: %d, end checkpoint: %d" % (begin_ckpt, config.eval.end_ckpt))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
        print("load from:", ckpt_filename)
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed and jax.host_id() == 0:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        import copy
        from models.utils import State

        new_state = checkpoints.restore_checkpoint(checkpoint_dir, target=None, step=ckpt)
        new_state.pop("optimizer")
        new_state["params_ema"] = jax.tree_map(lambda x, y: x.reshape(y.shape), new_state["params_ema"],
                                               state.params_ema.unfreeze())
        state = state.replace(**new_state)
        # Replicate the training state for executing on multiple devices
        pstate = flax.jax_utils.replicate(state)
        # Compute the loss function on the full evaluation dataset if loss computation is enabled
    
        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:
            state = jax.device_put(state)
            # Run sample generation for multiple rounds to create enough samples
            # Designed to be pre-emption safe. Automatically resumes when interrupted
            for r in range(begin_sampling_round, num_sampling_rounds):
                if jax.host_id() == 0:
                    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
                tf.io.gfile.makedirs(this_sample_dir)

                rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
                sample_rng = jnp.asarray(sample_rng)
                if config.sampling.method == "dpm_solver":
                    if config.sampling.return_intermediate:
                        samples, inter_samples, n = sampling_fn(sample_rng, pstate, steps=config.sampling.steps,
                                                                order=config.sampling.order,
                                                                return_intermediate=config.sampling.return_intermediate)
                    else:
                        samples, n = sampling_fn(sample_rng, pstate, steps=config.sampling.steps,
                                                 order=config.sampling.order,
                                                 return_intermediate=config.sampling.return_intermediate)
                else:
                    samples, n = sampling_fn(sample_rng, pstate)
                samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))

                if config.sampling.return_intermediate:
                    inter_samples = np.clip(inter_samples * 255., 0, 255).astype(np.uint8)
                    inter_samples = inter_samples.reshape(
                        (-1, samples.shape[0], config.data.image_size, config.data.image_size,
                         config.data.num_channels)).transpose((1, 0, 2, 3, 4))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())
                if config.sampling.return_intermediate:
                    with tf.io.gfile.GFile(
                            os.path.join(this_sample_dir, f"samples_inter_{r}.npz"), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, samples=inter_samples)
                        fout.write(io_buffer.getvalue())

                # Force garbage collection before calling TensorFlow code for Inception network
                gc.collect()
                latents = evaluation.run_inception_distributed(samples, inception_model,
                                                               inceptionv3=inceptionv3)
                if config.sampling.return_intermediate:
                    inter_latents = []
                    for i in range(inter_samples):
                        inter_latents.append(evaluation.run_inception_distributed(inter_samples[i], inception_model,
                                                                                  inceptionv3=inceptionv3))
                # Force garbage collection again before returning to JAX code
                gc.collect()
                # Save latent represents of the Inception network to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(
                        io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                    fout.write(io_buffer.getvalue())

                if config.sampling.return_intermediate:
                    for ii, sub_latents in enumerate(inter_latents):
                        with tf.io.gfile.GFile(
                                os.path.join(this_sample_dir, f"statistics_inter_{ii}_{r}.npz"), "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(
                                io_buffer, pool_3=sub_latents["pool_3"], logits=sub_latents["logits"])
                            fout.write(io_buffer.getvalue())

                # Update the intermediate evaluation state
                eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
                # Save an intermediate checkpoint directly if not the last round.
                # Otherwise save eval_meta after computing the Inception scores and FIDs
                if r < num_sampling_rounds - 1:
                    checkpoints.save_checkpoint(
                        eval_dir,
                        eval_meta,
                        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                        keep=1,
                        overwrite=True,
                        prefix=f"meta_{jax.host_id()}_")

            # Compute inception scores, FIDs and KIDs.
            if jax.host_id() == 0:
                # Load all statistics that have been previously computed and saved for each host
                all_logits = []
                all_pools = []
                for host in range(jax.host_count()):
                    this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")
                    stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
                    wait_message = False
                    while len(stats) < num_sampling_rounds:
                        if not wait_message:
                            logging.warning("Waiting for statistics on host %d" % (host,))
                            wait_message = True
                        stats = tf.io.gfile.glob(
                            os.path.join(this_sample_dir, "statistics_*.npz"))
                        time.sleep(30)
                    for stat_file in stats:
                        with tf.io.gfile.GFile(stat_file, "rb") as fin:
                            stat = np.load(fin)
                            if not inceptionv3:
                                all_logits.append(stat["logits"])
                            all_pools.append(stat["pool_3"])

                    if config.sampling.return_intermediate:
                        inter_total_all_logits = []
                        inter_total_all_pools = []
                        for ii in range(config.sampling.steps):
                            inter_all_logits = []
                            inter_all_pools = []
                            inter_stats = tf.io.gfile.glob(
                                os.path.join(this_sample_dir, f"statistics_inter_{ii}_*.npz"))
                            wait_message = False
                            while len(inter_stats) < num_sampling_rounds:
                                if not wait_message:
                                    logging.warning("Waiting for statistics on host %d" % (host,))
                                    wait_message = True
                                inter_stats = tf.io.gfile.glob(
                                    os.path.join(this_sample_dir, f"statistics_inter_{ii}_*.npz"))
                                time.sleep(30)
                            for stat_file in inter_stats:
                                with tf.io.gfile.GFile(stat_file, "rb") as fin:
                                    stat = np.load(fin)
                                    if not inceptionv3:
                                        inter_all_logits.append(stat["logits"])
                                    inter_all_pools.append(stat["pool_3"])
                            inter_total_all_logits.append(inter_all_logits)
                            inter_total_all_pools.append(inter_all_pools)
                if not inceptionv3:
                    all_logits = np.concatenate(
                        all_logits, axis=0)[:config.eval.num_samples]
                    if config.sampling.return_intermediate:
                        for ii in range(config.sampling.steps):
                            inter_total_all_logits[ii] = np.concatenate(
                                inter_total_all_logits[ii], axis=0)[:config.eval.num_samples]

                all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
                if config.sampling.return_intermediate:
                    for ii in range(config.sampling.steps):
                        inter_total_all_pools[ii] = np.concatenate(
                            inter_total_all_pools[ii], axis=0)[:config.eval.num_samples]

                # Load pre-computed dataset statistics.
                data_stats = evaluation.load_dataset_stats(config)
                data_pools = data_stats["pool_3"]

                # Compute FID/KID/IS on all samples together.
                if not inceptionv3:
                    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
                else:
                    inception_score = -1

                fid = tfgan.eval.frechet_classifier_distance_from_activations(
                    data_pools, all_pools)
                # Hack to get tfgan KID work for eager execution.
                tf_data_pools = tf.convert_to_tensor(data_pools)
                tf_all_pools = tf.convert_to_tensor(all_pools)
                kid = tfgan.eval.kernel_classifier_distance_from_activations(
                    tf_data_pools, tf_all_pools).numpy()
                del tf_data_pools, tf_all_pools

                logging.info(
                    "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                        ckpt, inception_score, fid, kid))
                with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                                       "wb") as f:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                    f.write(io_buffer.getvalue())

                if config.sampling.return_intermediate:
                    for ii in range(config.sampling.steps):
                        if not inceptionv3:
                            inception_score = tfgan.eval.classifier_score_from_logits(inter_total_all_logits[ii])
                        else:
                            inception_score = -1

                        fid = tfgan.eval.frechet_classifier_distance_from_activations(
                            data_pools, inter_total_all_pools[ii])
                        # Hack to get tfgan KID work for eager execution.
                        tf_data_pools = tf.convert_to_tensor(data_pools)
                        tf_all_pools = tf.convert_to_tensor(inter_total_all_pools[ii])
                        kid = tfgan.eval.kernel_classifier_distance_from_activations(
                            tf_data_pools, tf_all_pools).numpy()
                        del tf_data_pools, tf_all_pools

                        logging.info(
                            "ckpt-%d --- steps: %d, inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                                ckpt, ii + 1, inception_score, fid, kid))

            else:
                # For host_id() != 0.
                # Use file existence to emulate synchronization across hosts
                while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
                    time.sleep(1.)

            # Save eval_meta after computing IS/KID/FID to mark the end of evaluation for this checkpoint
            checkpoints.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                keep=1,
                prefix=f"meta_{jax.host_id()}_")

        else:
            # Skip sampling and save intermediate evaluation states for pre-emption
            eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
            checkpoints.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
                keep=1,
                prefix=f"meta_{jax.host_id()}_")

        begin_bpd_round = 0
        begin_sampling_round = 0

    # Remove all meta files after finishing evaluation
    meta_files = tf.io.gfile.glob(
        os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
    for file in meta_files:
        tf.io.gfile.remove(file)
