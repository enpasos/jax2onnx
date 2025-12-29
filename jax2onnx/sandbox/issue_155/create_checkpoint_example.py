#!/usr/bin/env python
# jax2onnx/sandbox/issue_155/create_checkpoint_example.py

# Import dependencies
import argparse
import logging
from pathlib import Path

# Import jax and brax
import jax
import jax.numpy as jp
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from ml_collections import config_dict
from absl import logging as absl_logging


# Main
def main():
    # Parse arguments
    args = parse_args()

    # Silence absl warnings
    absl_logging.set_verbosity(absl_logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # Resolve checkpoint path
    checkpoint_dir = Path(args.output).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    save_checkpoint(checkpoint_dir)


# Parse args
def parse_args():
    # Build parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to checkpoint directory")
    return parser.parse_args()


# Save checkpoint
def save_checkpoint(checkpoint_dir: Path, obs_dim: int = 39, action_dim: int = 6):
    # Hidden sizes
    hidden_sizes = [512, 256, 128]

    network_factory_kwargs = {
        "policy_hidden_layer_sizes": hidden_sizes,
        "value_hidden_layer_sizes": hidden_sizes,
    }
    networks = ppo_networks.make_ppo_networks(
        observation_size=obs_dim,
        action_size=action_dim,
        **network_factory_kwargs,
    )
    key_policy, key_value = jax.random.split(jax.random.PRNGKey(0))
    policy_params = networks.policy_network.init(key_policy)
    value_params = networks.value_network.init(key_value)

    # Running stats
    stats = running_statistics.init_state(jp.zeros((obs_dim,), dtype=jp.float32))

    # Config
    config = config_dict.ConfigDict()
    config.observation_size = obs_dim
    config.action_size = action_dim
    config.network_factory_kwargs = dict(network_factory_kwargs)
    config.normalize_observations = False

    # Save
    print(f"Saving checkpoint to {checkpoint_dir}")
    checkpoint.save(checkpoint_dir, 0, (stats, policy_params, value_params), config)


# Run main
if __name__ == "__main__":
    main()
