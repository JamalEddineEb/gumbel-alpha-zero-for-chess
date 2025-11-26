#!/usr/bin/env python3
"""Test symmetry transformations to ensure they work correctly."""

import numpy as np
import chess
import sys
sys.path.insert(0, '/home/jamal/projects/chess-endgame-mcts')

from src.utils.symmetries import transform_state, transform_policy, augment_sample
from src.utils.move_mapping import MoveMapping

def test_state_transform():
    """Test that state transformations preserve piece counts."""
    print("Testing state transformations...")
    
    # Create a simple state with a few pieces
    state = np.zeros((8, 8, 12))
    state[0, 0, 0] = 1  # White pawn at a8
    state[7, 7, 6] = 1  # Black pawn at h1
    state[3, 3, 4] = 1  # White queen at d5
    
    for transform_id in range(8):
        transformed = transform_state(state, transform_id)
        
        # Check that total piece count is preserved
        original_count = state.sum()
        transformed_count = transformed.sum()
        
        assert abs(original_count - transformed_count) < 1e-6, \
            f"Transform {transform_id}: piece count mismatch"
    
    print("✓ State transformations preserve piece counts")

def test_policy_transform():
    """Test that policy transformations preserve probability mass."""
    print("Testing policy transformations...")
    
    move_mapping = MoveMapping()
    
    # Create a simple policy with a few moves
    policy = np.zeros(move_mapping.num_actions)
    
    # Add some probability to a few moves
    policy[0] = 0.5  # a1a1 (self-move, but valid index)
    policy[63] = 0.3  # a1h1
    policy[100] = 0.2  # some other move
    
    for transform_id in range(8):
        transformed = transform_policy(policy, transform_id, move_mapping)
        
        # Check that total probability is preserved
        original_sum = policy.sum()
        transformed_sum = transformed.sum()
        
        assert abs(original_sum - transformed_sum) < 1e-6, \
            f"Transform {transform_id}: probability sum mismatch ({original_sum} vs {transformed_sum})"
    
    print("✓ Policy transformations preserve probability mass")

def test_augment_sample():
    """Test that augment_sample generates correct number of samples."""
    print("Testing sample augmentation...")
    
    move_mapping = MoveMapping()
    
    # Create dummy state and policy
    state = np.random.rand(8, 8, 12)
    policy = np.random.rand(move_mapping.num_actions)
    policy /= policy.sum()
    
    # Generate augmented samples
    augmented = augment_sample(state, policy, move_mapping, num_augmentations=8)
    
    assert len(augmented) == 8, f"Expected 8 samples, got {len(augmented)}"
    
    for i, (aug_state, aug_policy) in enumerate(augmented):
        assert aug_state.shape == (8, 8, 12), f"Sample {i}: wrong state shape"
        assert aug_policy.shape == policy.shape, f"Sample {i}: wrong policy shape"
        assert abs(aug_policy.sum() - 1.0) < 1e-5, f"Sample {i}: policy doesn't sum to 1"
    
    print("✓ Sample augmentation generates correct number of samples")

if __name__ == "__main__":
    print("Running symmetry transformation tests...\n")
    
    try:
        test_state_transform()
        test_policy_transform()
        test_augment_sample()
        
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
