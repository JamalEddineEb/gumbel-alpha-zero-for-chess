import numpy as np
import chess


def flip_horizontal(state):
    """Flip board state horizontally (mirror along vertical axis)."""
    return np.flip(state, axis=1)


def flip_vertical(state):
    """Flip board state vertically (mirror along horizontal axis)."""
    return np.flip(state, axis=0)


def rotate_90(state):
    """Rotate board state 90 degrees clockwise."""
    return np.rot90(state, k=-1, axes=(0, 1))


def rotate_180(state):
    """Rotate board state 180 degrees."""
    return np.rot90(state, k=2, axes=(0, 1))


def rotate_270(state):
    """Rotate board state 270 degrees clockwise."""
    return np.rot90(state, k=-3, axes=(0, 1))


def get_square_transform(transform_id):
    """
    Get the square transformation function for a given transform ID.
    
    Args:
        transform_id: 0-7 representing different symmetries
        
    Returns:
        Function that transforms a square index (0-63)
    """
    def identity(sq):
        return sq
    
    def flip_h(sq):
        rank, file = divmod(sq, 8)
        return rank * 8 + (7 - file)
    
    def flip_v(sq):
        rank, file = divmod(sq, 8)
        return (7 - rank) * 8 + file
    
    def rot_90(sq):
        rank, file = divmod(sq, 8)
        return file * 8 + (7 - rank)
    
    def rot_180(sq):
        rank, file = divmod(sq, 8)
        return (7 - rank) * 8 + (7 - file)
    
    def rot_270(sq):
        rank, file = divmod(sq, 8)
        return (7 - file) * 8 + rank
    
    def flip_h_rot_90(sq):
        return rot_90(flip_h(sq))
    
    def flip_v_rot_90(sq):
        return rot_90(flip_v(sq))
    
    transforms = [
        identity,      # 0: no transform
        flip_h,        # 1: horizontal flip
        flip_v,        # 2: vertical flip
        rot_90,        # 3: 90° rotation
        rot_180,       # 4: 180° rotation
        rot_270,       # 5: 270° rotation
        flip_h_rot_90, # 6: horizontal flip + 90° rotation
        flip_v_rot_90, # 7: vertical flip + 90° rotation
    ]
    
    return transforms[transform_id]


def transform_state(state, transform_id):
    """
    Apply symmetry transformation to board state.
    
    Args:
        state: (8, 8, 12) numpy array
        transform_id: 0-7 representing different symmetries
        
    Returns:
        Transformed state
    """
    if transform_id == 0:
        return state
    elif transform_id == 1:
        return flip_horizontal(state)
    elif transform_id == 2:
        return flip_vertical(state)
    elif transform_id == 3:
        return rotate_90(state)
    elif transform_id == 4:
        return rotate_180(state)
    elif transform_id == 5:
        return rotate_270(state)
    elif transform_id == 6:
        return rotate_90(flip_horizontal(state))
    elif transform_id == 7:
        return rotate_90(flip_vertical(state))
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")


def transform_policy(policy, transform_id, move_mapping):
    """
    Apply symmetry transformation to policy vector.
    
    Args:
        policy: (4096+,) numpy array of move probabilities
        transform_id: 0-7 representing different symmetries
        move_mapping: MoveMapping instance
        
    Returns:
        Transformed policy vector
    """
    if transform_id == 0:
        return policy
    
    transformed_policy = np.zeros_like(policy)
    square_transform = get_square_transform(transform_id)
    
    # For base moves (first 4096 indices): from_sq * 64 + to_sq
    for from_sq in range(64):
        for to_sq in range(64):
            idx = from_sq * 64 + to_sq
            if policy[idx] > 0:
                # Transform squares
                new_from = square_transform(from_sq)
                new_to = square_transform(to_sq)
                new_idx = new_from * 64 + new_to
                transformed_policy[new_idx] = policy[idx]
    
    # For promotion moves (indices >= 4096)
    # These are more complex, so we'll handle them via UCI
    for idx in range(4096, len(policy)):
        if policy[idx] > 0:
            try:
                uci = move_mapping.get_move_uci(idx)
                if uci is None:
                    continue
                
                # Parse move
                from_sq = chess.SQUARE_NAMES.index(uci[:2])
                to_sq = chess.SQUARE_NAMES.index(uci[2:4])
                
                # Transform squares
                new_from = square_transform(from_sq)
                new_to = square_transform(to_sq)
                
                # Handle promotion
                promotion = uci[4:] if len(uci) > 4 else ''
                
                # Create new UCI
                new_uci = chess.SQUARE_NAMES[new_from] + chess.SQUARE_NAMES[new_to] + promotion
                
                # Get new index
                new_idx = move_mapping.get_index(new_uci)
                if new_idx is not None:
                    transformed_policy[new_idx] = policy[idx]
            except (KeyError, IndexError, ValueError):
                # Skip invalid moves
                continue
    
    # Renormalize
    total = transformed_policy.sum()
    if total > 0:
        transformed_policy /= total
    
    return transformed_policy


def augment_sample(state, policy, move_mapping, num_augmentations=8):
    """
    Generate augmented samples using symmetries.
    
    Args:
        state: (8, 8, 12) board state
        policy: (4096,) policy vector
        move_mapping: MoveMapping instance
        num_augmentations: Number of augmentations (1-8)
        
    Returns:
        List of (augmented_state, augmented_policy) tuples
    """
    samples = []
    
    for transform_id in range(min(num_augmentations, 8)):
        aug_state = transform_state(state, transform_id)
        aug_policy = transform_policy(policy, transform_id, move_mapping)
        samples.append((aug_state, aug_policy))
    
    return samples
