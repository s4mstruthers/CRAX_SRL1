"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import functools
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import wandb
from crax import envs
from training.config import build_base_parser
from training.run_utils import (
    collect_rollout_metrics, record_episode_video, setup_gpu_environment,
    get_algorithm_train_fn, filter_kwargs_for_fn, custom_progress_fn,
    make_vision_network_factory, morphology_override, VISION_CAMERA_OVERRIDES,
    make_periodic_vision_video_fn,
)
from crax.envs.limb_colors import colorize_env_limbs


def main():
    """Main function to run training from command line."""
    parser = build_base_parser(description='Train Safe-Brax agents from config files')
    config = parser.parse_args()

    env_name = config.env_name
    alg_name = config.alg
    difficulty = config.difficulty
    use_wandb = config.use_wandb

    # Fill in the morphology-specific pixel-obs training camera, but only if
    # the user didn't explicitly pass --vision_camera
    if config.vision_camera is None:
        config.vision_camera = morphology_override(env_name, VISION_CAMERA_OVERRIDES) or 'vision'

    # Setup GPU environment
    setup_gpu_environment(vision=config.vision)

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_name = f"{env_name}_Level_{difficulty}_{alg_name}_seed{seed}_{timestamp}"

        # Build vision kwargs. Pixel-obs wrapping happens inside the training function
        vision_kwargs = None
        if config.vision:
            vision_kwargs = dict(
                camera=config.vision_camera,
                height=config.vision_height,
                width=config.vision_width,
                obs_mode=config.vision_obs_mode,
                frame_stack=config.vision_frame_stack,
            )
            print(
                f"Vision mode: GPU rendering (MJWarp), "
                f"camera='{config.vision_camera}', "
                f"{config.vision_width}x{config.vision_height}"
            )

        # Create environments with a difficulty level
        env_kwargs = config.env_kwargs or {}
        if env_name == 'safe_velocity':
            env_kwargs['agent'] = config.agent
        if config.vision:
            # GpuPixelObservationWrapper reads geom_xpos/cam_xpos, which only the MJX pipeline populates.
            env_kwargs.setdefault('backend', 'mjx')
        env = envs.get_environment(env_name, level=difficulty, **env_kwargs)
        eval_env = envs.get_environment(env_name, level=difficulty, **env_kwargs)

        # Distinct per-limb colours in the pixel observations
        if config.vision and config.vision_limb_colors:
            n_colored = colorize_env_limbs(env, env_name)
            colorize_env_limbs(eval_env, env_name)
            if n_colored:
                print(f"Vision mode: recoloured {n_colored} limb geoms for pixel observations.")
            else:
                print(f"Vision mode: --vision_limb_colors set but '{env_name}' has no limb "
                      f"colour scheme (see crax/envs/limb_colors.py); left unchanged.")

        # Determine the episode length
        episode_length = config.episode_length or env_kwargs.get('episode_length') or getattr(env, 'episode_length', None)

        # Periodic mid-training video, --vision only: a dedicated single-env vision-wrapped
        # rollout env, reading frames straight off its own GPU (MJWarp) pixel observations.
        video_fn = None
        if config.vision and not config.skip_video:
            video_env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'episode_length'}
            # Re-colour inside `pre_vision_fn`, not on the returned env
            use_limb_colors = config.vision and config.vision_limb_colors
            periodic_video_env = envs.create(
                env_name, level=difficulty,
                episode_length=episode_length,
                auto_reset=True,
                batch_size=1,
                vision=True,
                vision_kwargs=dict(**vision_kwargs, num_envs=1),
                pre_vision_fn=(lambda e: colorize_env_limbs(e, env_name)) if use_limb_colors else None,
                **video_env_kwargs,
            )
            video_fn = make_periodic_vision_video_fn(
                periodic_video_env,
                every_steps=config.video_every_steps,
                steps=config.periodic_video_steps,
                num_episodes=config.num_video_episodes,
                pixel_camera=config.vision_camera,
                # Same extra-camera set vector-obs training's end-of-run video
                # uses (config.cameras, default ["fixedfar", "vision"]) minus
                # whichever one is already pixel_camera's own clip.
                extra_cameras=[c for c in config.cameras if c != config.vision_camera],
                frame_stack=config.vision_frame_stack,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                run_name=run_name,
                deterministic=config.deterministic_eval,
                log_to_wandb=config.use_wandb,
                seed=seed,
            )

        print(f"Training environment '{env_name}' instantiated with difficulty {difficulty}.")
        print(f"Evaluation environment '{env_name}' instantiated with difficulty {difficulty}.")

        cli_cfg = vars(config)
        runtime_cfg = {"seed": seed, "timestamp": timestamp, "episode_length": episode_length}
        cfg = {**cli_cfg, **runtime_cfg}

        if use_wandb:
            # Prepare wandb config
            wandb_config = cfg.copy()
            wandb_project = config.wandb_project
            wandb_group = config.wandb_group if config.wandb_group else env_name
            wandb_tags = config.wandb_tags

            # Initialize wandb
            wandb.init(
                project=wandb_project,
                name=run_name,
                id=run_name,
                config=wandb_config,
                group=wandb_group,
                job_type=alg_name,
                tags=wandb_tags,
            )

        if config.store_model:
            root_dir = Path(__file__).parent.parent.resolve()  # repo root, not training/
            ckpt_root = root_dir / config.model_dir / run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        # Setup metrics collection
        progress_fn = functools.partial(custom_progress_fn, use_wandb=use_wandb, verbose=not config.quiet)

        # Get the appropriate training function
        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

        # Budget used by the episode-level safety metrics ('safety_ep/frac_over_budget').
        # The safe algorithms do not forward extra kwargs to ppo/train.py, so it is passed
        # through the environment instead.
        os.environ['CRAX_SAFETY_BOUND'] = str(config.safety_bound)

        # Plain PPO ignores cost when learning, and PPO-Saute folds it into the reward,
        # so neither collects the per-step cost by default. Record it anyway so they get
        # the same per-update 'safety/' metrics as the other safe algorithms (logging
        # only; learning is unchanged). Only safe_* environments provide a cost signal.
        if alg_name in ('ppo', 'ppo_cost', 'ppo_saute') and env_name.startswith('safe_'):
            train_kwargs['extra_fields'] = (
                'truncation', 'episode_metrics', 'episode_done', 'cost')

        # Inject vision network factory + pixel-obs wrapping kwargs if vision mode is enabled
        if config.vision:
            state_obs_key = 'state' if config.vision_obs_mode == 'pixels+state' else ''
            train_kwargs['network_factory'] = make_vision_network_factory(
                alg_name,
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            train_kwargs['augment_pixels'] = config.vision_augment
            train_kwargs['vision_kwargs'] = vision_kwargs
            if video_fn is not None:
                train_kwargs['policy_params_fn'] = video_fn
            train_kwargs = filter_kwargs_for_fn(train_fn_base, train_kwargs)
            if 'vision_kwargs' not in train_kwargs:
                raise ValueError(
                    f"--vision was set but algorithm '{alg_name}' does not "
                    f"support pixel observations (its train() has no "
                    f"'vision_kwargs' parameter)."
                )

        # Create the training function
        train_fn = functools.partial(train_fn_base, **train_kwargs)

        # Train the agent
        make_inference_fn, params, final_metrics, eval_env = train_fn(
            environment=env,
            eval_env=eval_env,
            progress_fn=progress_fn
        )
        print("Training finished.")

        # Log final metrics to the wandb run summary.
        # final_metrics mixes scalars with per-update arrays from the last training epoch
        # (JAX arrays, not only NumPy), and some entries can be NaN (e.g. an update in which
        # no episode finished). Arrays are reduced with a NaN-aware mean and non-finite values
        # are dropped, because wandb crashes when it tries to histogram an array with NaN.
        # The summary is used instead of wandb.log: the per-update history is already logged,
        # and a history row at step=num_timesteps would be out of order (training rounds the
        # step count up) and would mix averaged values into the per-update curves.
        if use_wandb and wandb.run is not None and final_metrics:
            final_log_data = {}
            for key, value in final_metrics.items():
                if value is None:
                    continue
                try:
                    arr = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if arr.size == 0 or not np.isfinite(arr).any():
                    continue
                final_log_data[f"final/{key}"] = float(np.nanmean(arr)) if arr.ndim > 0 else float(arr)
            if final_log_data:
                wandb.run.summary.update(final_log_data)

        if not config.skip_rollout:
            print(f"\nPerforming rollout evaluation...")
            rollout_metrics = collect_rollout_metrics(
                env_name=env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                level=config.difficulty,
                env_kwargs=config.env_kwargs,
            )

        if not config.skip_video:
            if config.vision:
                # Force one last clip from the actual final params
                video_fn(int(config.num_timesteps), make_inference_fn, params, force=True)
            else:
                video_length = config.video_length if config.video_length else config.episode_length
                if video_length is None:
                    video_length = getattr(eval_env, 'episode_length', None) or getattr(eval_env, 'default_episode_length', None)
                video_env = envs.get_environment(
                    env_name, level=difficulty, **env_kwargs,
                )
                record_episode_video(
                    env=video_env,
                    make_inference_fn=make_inference_fn,
                    params=params,
                    steps=video_length,
                    cameras=config.cameras,
                    width=config.video_width,
                    height=config.video_height,
                    fps=config.video_fps,
                    frame_stride=config.video_frame_stride,
                    out_name=run_name,
                    log_to_wandb=config.use_wandb,
                    seed=seed,
                    num_episodes=config.num_video_episodes,
                )

        # Finish wandb run if active
        if config.use_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
