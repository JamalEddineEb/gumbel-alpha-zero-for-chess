import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque
import chess
import numpy as np

from src.mcts_agent import MCTSAgent
from src.environment import ChessEnv
from src.utils.symmetries import augment_sample

import argparse

def train_agent(start_fen=None):
    # Training parameters
    n_episodes = 10000
    n_simulations = 100 
    batch_size = 256

    # Initialize environment and agent
    env = ChessEnv(demo_mode=False)
    state_size = (8 , 8 , 12)  # 8x8 board with 12 channels
    mates = 0
    agent = MCTSAgent(state_size=state_size, n_simulations=n_simulations)
    
    episodes = n_episodes
    checkpoint_frequency = 1 # Train every episode
    target_update_frequency = 10  # Update target model less frequently for stability

    import tensorflow as tf
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_file = "model_checkpoint.weights.h5"

    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        try:
            agent.load(model_file)
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with random weights.")
    else:
        print("No model found, training a new one.")


    # Pure self-play (AlphaZero style) - no curriculum
    # Agent learns from random positions, exploring diverse scenarios
    recent_results = deque(maxlen=100)  # Track last 100 games for monitoring

    for e in range(episodes):
        # Always use random positions (difficulty=2) unless FEN is provided
        env.reset(fen=start_fen, difficulty=2 if not start_fen else None)
        
        moves_made = 0
        game_samples = []  # list of (state, improved_policy, player_color)

        print(f"episode {e}")

        max_moves = 50  # prevent endless shuffling

        while not env.done and moves_made < max_moves:
            state = env.get_state()
            current_player = env.board.turn

            # Run search at the root
            move, improved_policy, v_pi = agent.simulate(env)

            if move is None:
                # No legal moves
                break

            # Store (s, π′, player) for this position
            game_samples.append((state, improved_policy, current_player))

            # Play move
            _, reward, done = env.step(move)
            moves_made += 1
            mates += reward

            # print(f"Move {moves_made}: {move}, done={done}")
            # print(env.board.unicode())

        # ----- game finished or max_moves reached -----
        # Use final game result as value target z
        result = env.board.result(claim_draw=True)  # "1-0","0-1","1/2-1/2","*"
        
        z_white = 0.0
        if result == "1-0":
            z_white = 1.0
            recent_results.append(1) # Win for White
        elif result == "0-1":
            z_white = -1.0
            recent_results.append(0) # Loss for White (shouldn't happen in QK vs K)
        elif result == "1/2-1/2":
            z_white = 0.0
            recent_results.append(0) # Draw
        else:
            # game truncated or unknown
            z_white = 0.0
            recent_results.append(0)

        print("z_white", z_white)

        # Push all (state, π′, z) into replay memory with augmentation
        for s, pi, player in game_samples:
            if player == chess.WHITE:
                z = z_white
            else:
                z = -z_white
            
            # Generate augmented samples (8 symmetries)
            augmented = augment_sample(s, pi, agent.move_mapping, num_augmentations=8)
            for aug_state, aug_policy in augmented:
                agent.memory.append((aug_state, aug_policy, z))


        # Monitor win rate (for tracking progress, not for curriculum)
        if len(recent_results) >= 50:
            win_rate = sum(recent_results) / len(recent_results)
            print(f"Recent Win Rate: {win_rate:.2f}")


        print(
            f"Episode: {e}/{episodes}, "
            f"moves: {moves_made}, "
            f"mates: {mates}/{e+1}, "
            f"final result: {result}"
        )

        # --- TRAINING STEP -----------------------------------------
        if len(agent.memory) >= batch_size and e % checkpoint_frequency == 0:
            agent.replay(batch_size)
            agent.save(model_file)

        if e % target_update_frequency == 0:
            agent.update_target_model()

    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Chess MCTS Agent')
    parser.add_argument('--fen', type=str, help='Starting FEN string')
    parser.add_argument('--pgn', type=str, help='Path to PGN file to load starting position from')
    
    args = parser.parse_args()
    
    start_fen = args.fen
    
    if args.pgn:
        with open(args.pgn) as f:
            game = chess.pgn.read_game(f)
            if game:
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                start_fen = board.fen()
                print(f"Loaded position from PGN: {start_fen}")
            else:
                print("Could not read PGN file.")

    agent = train_agent(start_fen=start_fen)