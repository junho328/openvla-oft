"""
Dimension validation test for ACPPO implementation.
Run this to verify all tensor dimensions are correct.
"""

import torch
import numpy as np
from config import ACPPOConfig, TWOARM_ACPPO_CONSTANTS


def test_config_dimensions():
    """Test that config dimensions are consistent."""
    cfg = ACPPOConfig()
    
    # Action distribution dimension
    expected_action_dist_dim = cfg.action_dim * cfg.num_actions_chunk * 2  # mu + sigma
    assert cfg.action_dist_dim == expected_action_dist_dim, \
        f"action_dist_dim mismatch: {cfg.action_dist_dim} != {expected_action_dist_dim}"
    
    # Extended proprio dimension
    expected_proprio_dim_agent1 = cfg.proprio_dim_agent0 + cfg.action_dist_dim
    assert cfg.proprio_dim_agent1 == expected_proprio_dim_agent1, \
        f"proprio_dim_agent1 mismatch: {cfg.proprio_dim_agent1} != {expected_proprio_dim_agent1}"
    
    print("✅ Config dimensions are correct")
    print(f"   - action_dim: {cfg.action_dim}")
    print(f"   - num_actions_chunk: {cfg.num_actions_chunk}")
    print(f"   - action_dist_dim: {cfg.action_dist_dim} (= {cfg.action_dim} × {cfg.num_actions_chunk} × 2)")
    print(f"   - proprio_dim_agent0: {cfg.proprio_dim_agent0}")
    print(f"   - proprio_dim_agent1: {cfg.proprio_dim_agent1} (= {cfg.proprio_dim_agent0} + {cfg.action_dist_dim})")


def test_buffer_dimensions():
    """Test rollout buffer storage dimensions."""
    from rollout_buffer import MultiAgentRolloutBufferACPPO
    
    buffer = MultiAgentRolloutBufferACPPO(
        buffer_size=10,
        num_agents=2,
        num_envs=1,
        action_dim=6,
        action_chunk_size=4,
        proprio_dim=8,
    )
    
    # Check storage shapes
    assert len(buffer.action_means) == 2, "Should have action_means for 2 agents"
    assert buffer.action_means[0].shape == (10, 1, 24), \
        f"action_means shape mismatch: {buffer.action_means[0].shape} != (10, 1, 24)"
    
    assert len(buffer.action_stds) == 2, "Should have action_stds for 2 agents"
    assert buffer.action_stds[0].shape == (10, 1, 24), \
        f"action_stds shape mismatch: {buffer.action_stds[0].shape} != (10, 1, 24)"
    
    assert len(buffer.values) == 2, "Should have values for 2 agents"
    assert buffer.values[0].shape == (10, 1), \
        f"values shape mismatch: {buffer.values[0].shape} != (10, 1)"
    
    print("✅ Buffer dimensions are correct")
    print(f"   - action_means[agent]: {buffer.action_means[0].shape}")
    print(f"   - action_stds[agent]: {buffer.action_stds[0].shape}")
    print(f"   - values[agent]: {buffer.values[0].shape}")


def test_extended_proprio():
    """Test extended proprio creation."""
    from observation_utils import create_extended_proprio
    
    proprio = np.random.randn(8).astype(np.float32)
    action_mean = np.random.randn(24).astype(np.float32)
    action_std = np.abs(np.random.randn(24)).astype(np.float32)
    
    extended = create_extended_proprio(proprio, action_mean, action_std)
    
    assert extended.shape == (56,), f"Extended proprio shape mismatch: {extended.shape} != (56,)"
    assert extended.dtype == np.float32, f"Extended proprio dtype mismatch: {extended.dtype} != float32"
    
    # Verify concatenation
    assert np.allclose(extended[:8], proprio), "Proprio part mismatch"
    assert np.allclose(extended[8:32], action_mean), "Action mean part mismatch"
    assert np.allclose(extended[32:56], action_std), "Action std part mismatch"
    
    print("✅ Extended proprio dimensions are correct")
    print(f"   - Input proprio: {proprio.shape}")
    print(f"   - Input action_mean: {action_mean.shape}")
    print(f"   - Input action_std: {action_std.shape}")
    print(f"   - Output extended: {extended.shape}")


def test_gae_computation():
    """Test GAE computation logic with mock data."""
    from rollout_buffer import MultiAgentRolloutBufferACPPO
    
    buffer = MultiAgentRolloutBufferACPPO(
        buffer_size=5,
        num_agents=2,
        num_envs=1,
        action_dim=6,
        action_chunk_size=4,
        gamma_prime=0.99,
        lambda_prime=0.95,
    )
    
    # Fill buffer with mock data
    for t in range(5):
        buffer.add(
            values=[np.array([t * 0.1]), np.array([t * 0.2])],  # Per-agent values
            reward=np.array([1.0]),
            done=np.array([0.0]),
        )
    buffer.full = True
    
    # Compute returns and advantages
    last_values = [np.array([0.5]), np.array([0.6])]
    last_dones = np.array([0.0])
    
    buffer.compute_returns_and_advantages(last_values, last_dones)
    
    # Verify shapes
    assert len(buffer.advantages) == 2, "Should have advantages for 2 agents"
    assert buffer.advantages[0].shape == (5, 1), \
        f"Advantages shape mismatch: {buffer.advantages[0].shape} != (5, 1)"
    
    assert len(buffer.returns) == 2, "Should have returns for 2 agents"
    assert buffer.returns[0].shape == (5, 1), \
        f"Returns shape mismatch: {buffer.returns[0].shape} != (5, 1)"
    
    print("✅ GAE computation dimensions are correct")
    print(f"   - advantages[agent]: {buffer.advantages[0].shape}")
    print(f"   - returns[agent]: {buffer.returns[0].shape}")
    
    # Print sample values for inspection
    print(f"   - Sample advantages[0]: {buffer.advantages[0].flatten()[:3]}...")
    print(f"   - Sample advantages[1]: {buffer.advantages[1].flatten()[:3]}...")


def test_action_distribution_tensor_shapes():
    """Test tensor shapes for action distribution processing."""
    batch_size = 4
    action_dim = 6
    chunk_size = 4
    proprio_dim = 8
    
    # Simulated shapes
    mu = torch.randn(batch_size, action_dim * chunk_size)  # (4, 24)
    sigma = torch.randn(batch_size, action_dim * chunk_size)  # (4, 24)
    proprio = torch.randn(batch_size, proprio_dim)  # (4, 8)
    
    # Create extended proprio
    proprio_extended = torch.cat([proprio, mu, sigma], dim=-1)  # (4, 56)
    
    assert proprio_extended.shape == (batch_size, 56), \
        f"Extended proprio shape mismatch: {proprio_extended.shape} != ({batch_size}, 56)"
    
    print("✅ Action distribution tensor shapes are correct")
    print(f"   - mu: {mu.shape}")
    print(f"   - sigma: {sigma.shape}")
    print(f"   - proprio: {proprio.shape}")
    print(f"   - proprio_extended: {proprio_extended.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("ACPPO Dimension Validation Tests")
    print("=" * 60)
    print()
    
    test_config_dimensions()
    print()
    
    test_buffer_dimensions()
    print()
    
    test_extended_proprio()
    print()
    
    test_gae_computation()
    print()
    
    test_action_distribution_tensor_shapes()
    print()
    
    print("=" * 60)
    print("All dimension tests passed! ✅")
    print("=" * 60)
