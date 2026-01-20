"""Check initial distance between peg and hole in TwoArmPegInHole environment."""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.twoarmpeginhole.twoarm_utils import get_twoarm_env

def check_initial_distance():
    """Check the initial distance between peg and hole."""
    env, _ = get_twoarm_env(
        model_family="openvla",
        resolution=256,
        robot1="Panda",
        robot2="Panda",
        controller="BASIC",
        env_configuration="opposed",
        reward_shaping=True,
    )
    
    # Reset environment
    obs = env.reset()
    
    # Get initial distance from info
    dummy_action = [0.0] * 12
    obs, reward, done, info = env.step(dummy_action)
    
    initial_dist = info.get("reward/peg_hole_dist", None)
    
    print(f"Initial peg-hole distance: {initial_dist:.4f} meters")
    print(f"Initial peg-hole distance: {initial_dist * 100:.2f} cm")
    
    # Check a few more steps to see if distance changes
    print("\nDistance over first 5 steps:")
    for i in range(5):
        obs, reward, done, info = env.step(dummy_action)
        dist = info.get("reward/peg_hole_dist", None)
        print(f"Step {i+1}: {dist:.4f} m ({dist * 100:.2f} cm)")
    
    return initial_dist

if __name__ == "__main__":
    initial_dist = check_initial_distance()
    print(f"\nRecommended max_peg_hole_distance: {initial_dist * 1.5:.3f} m (1.5x initial distance)")
