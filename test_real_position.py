#!/usr/bin/env python3
"""Verify symmetry transformations with actual chess positions."""

import numpy as np
import chess
import sys
sys.path.insert(0, '/home/jamal/projects/chess-endgame-mcts')

from src.environment import ChessEnv
from src.utils.symmetries import transform_state, augment_sample
from src.utils.move_mapping import MoveMapping

def test_real_position():
    """Test with a real chess position."""
    print("Testing with real K+Q vs K position...")
    
    env = ChessEnv(demo_mode=False)
    env.reset(difficulty=2)  # Random position
    
    state = env.get_state()
    move_mapping = MoveMapping()
    
    # Create a dummy policy (uniform over all moves)
    policy = np.ones(move_mapping.num_actions) / move_mapping.num_actions
    
    # Generate augmented samples
    augmented = augment_sample(state, policy, move_mapping, num_augmentations=8)
    
    print(f"Original position:")
    print(env.board.unicode())
    print(f"\nOriginal state sum: {state.sum()}")
    print(f"Original policy sum: {policy.sum():.6f}")
    
    for i, (aug_state, aug_policy) in enumerate(augmented):
        print(f"\nAugmentation {i}:")
        print(f"  State sum: {aug_state.sum()}")
        print(f"  Policy sum: {aug_policy.sum():.6f}")
        
        # Verify piece count is preserved
        assert abs(state.sum() - aug_state.sum()) < 1e-6, \
            f"Augmentation {i}: piece count mismatch"
        
        # Verify policy sums to 1
        assert abs(aug_policy.sum() - 1.0) < 1e-5, \
            f"Augmentation {i}: policy doesn't sum to 1"
    
    print("\n✅ All augmentations preserve piece counts and probability mass!")

if __name__ == "__main__":
    test_real_position()
