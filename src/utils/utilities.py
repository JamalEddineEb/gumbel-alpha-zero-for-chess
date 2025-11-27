import numpy as np
import math

from src.mcts_node import MCTSNode



def sample_gumbel(shape, eps=1e-9):
    # U ~ Uniform(0,1) clipped for stability
    U = np.random.uniform(low=eps, high=1.0 - eps, size=shape)
    return -np.log(-np.log(U))



def gumbel_top_k_root_candidates(root, k):
    # Gumbel-Top-k scores
    g = sample_gumbel(len(root.children))

    scores = {}
    for i, (move, child) in enumerate(root.children.items()):
        # Add epsilon to avoid log(0)
        logits = np.log(child.prior + 1e-9)
        scores[child] = logits + g[i]

    sorted_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_k_children = [child for child, _ in sorted_moves[:k]]

    return top_k_children


def select_child_sequential_policy(node):
    """
    Non-root selection using Equation 14:
    argmax_a (π′(a) - N(a) / (1 + Σ_b N(b)))
    where π′(a) is the policy prior (stored in node.priors_cache).
    """
    total_visits = sum(ch.visits for ch in node.children.values())

    denom = 1.0 + total_visits
    
    best_score = -1e9
    for child in node.children.values():
        n = 0 if child is None else child.visits
        score = child.prior - (n / denom)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child



def ceil_log2(x):
    return math.ceil(math.log2(max(1, x)))

def backup_path(path, leaf_value):
    v = leaf_value
    for n in reversed(path):
        n.visits += 1
        n.value += v
        v = -v



        
