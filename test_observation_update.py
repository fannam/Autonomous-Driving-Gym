#!/usr/bin/env python3
"""Quick test to verify DetailedOccupancyGrid updates correctly between steps."""

import sys
sys.path.insert(0, "source/algorithm/PPO-based/PPO-traditional")

import numpy as np
from PPOTraditional.environment.config import init_env

def test_observation_updates():
    """Test that observations change as environment evolves."""
    print("Testing DetailedOccupancyGrid observation updates...\n")

    # Create environment
    env = init_env(seed=42, stage="train", config_path=None, render_mode=None)

    # Reset and get initial observation
    obs1, info1 = env.reset(seed=42)
    print(f"Initial observation shape: {obs1.shape}")
    print(f"Initial observation stats:")
    for i, feature in enumerate(["presence", "speed", "on_lane", "on_road"]):
        channel = obs1[i]
        print(f"  {feature}: min={channel.min():.3f}, max={channel.max():.3f}, "
              f"mean={channel.mean():.3f}, nonzero={np.count_nonzero(channel)}")

    # Take some random actions
    print(f"\nTaking 5 steps with action=4 (SLOWER)...")
    for step in range(5):
        obs_next, reward, terminated, truncated, info = env.step(4)  # SLOWER action
        print(f"  Step {step+1}: reward={reward:.3f}, speed={info.get('forward_speed', 0):.2f} m/s")

    obs2 = obs_next
    print(f"\nObservation after 5 SLOWER steps:")
    for i, feature in enumerate(["presence", "speed", "on_lane", "on_road"]):
        channel = obs2[i]
        print(f"  {feature}: min={channel.min():.3f}, max={channel.max():.3f}, "
              f"mean={channel.mean():.3f}, nonzero={np.count_nonzero(channel)}")

    # Check if observations changed
    obs_diff = np.abs(obs2 - obs1).sum()
    print(f"\nTotal absolute difference between obs1 and obs2: {obs_diff:.2f}")

    if obs_diff < 1e-6:
        print("❌ WARNING: Observations did NOT change! Possible update issue.")
    else:
        print("✅ Observations changed correctly between steps.")

    # Test presence channel specifically (should show vehicles moving)
    presence_diff = np.abs(obs2[0] - obs1[0]).sum()
    print(f"\nPresence channel difference: {presence_diff:.2f}")
    if presence_diff < 1e-6:
        print("❌ WARNING: Vehicle presence did NOT change!")
    else:
        print("✅ Vehicle positions updated correctly.")

    env.close()
    return obs_diff > 1e-6

if __name__ == "__main__":
    success = test_observation_updates()
    sys.exit(0 if success else 1)
