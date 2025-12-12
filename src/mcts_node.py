import math
import numpy as np

from src.utils.move_mapping import MoveMapping

class MCTSNode:
    slots = ("parent","children","visits","value","prior","move","expanded")
    def __init__(self,prior=0,move=None):
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.move = move
        self.expanded = False


    def expand(self, legal_moves, priors):
        for move in legal_moves:
            if move not in self.children:
                self.children[move] = MCTSNode(prior=priors.get(move,0.0),move=move)

    def expand_leaf(self, env, model):
        self.expanded = True
        
        state = env.get_state()
        # Predict using the model (call directly for speed/stability instead of .predict)
        # Output is [policy_probs, value]
        policy_probs, value = model(state[None], training=False)
        
        # Convert tensors to numpy
        policy_probs = policy_probs.numpy()[0]
        value = float(value.numpy()[0])
        
        # Mask illegal moves
        legal_moves = list(env.get_legal_actions()) # Changed from env.board.legal_moves to env.get_legal_actions() to match original logic
        
        # Create children
        move_mapping = MoveMapping()
        
        # Prepare priors for the expand method
        priors = {}
        total_p = 0.0
        for move in legal_moves:
            uci = move.uci()
            idx = move_mapping.get_index(uci)
            p = max(1e-12, policy_probs[idx]) # Ensure prior is not zero
            priors[move] = p
            total_p += p
        
        # Normalize priors
        for move in legal_moves:
            priors[move] /= total_p

        self.expand(legal_moves, priors) 
        return value

    def update(self, value):
        self.visits += 1
        self.value += value