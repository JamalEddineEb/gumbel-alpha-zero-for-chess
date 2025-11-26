import multiprocessing as mp
import chess
import numpy as np
from collections import deque

from src.mcts_agent import MCTSAgent
from src.environment import ChessEnv


def play_game_worker(worker_id, weight_queue, result_queue, n_simulations, start_fen, difficulty, max_moves):
    """
    Worker function that plays a single game and returns samples.
    
    Args:
        worker_id: Unique identifier for this worker
        weight_queue: Queue to receive model weights from main process
        result_queue: Queue to send game samples back to main process
        n_simulations: Number of MCTS simulations per move
        start_fen: Starting FEN position (or None for random)
        difficulty: Difficulty level for position generation
        max_moves: Maximum moves per game
    
    Returns:
        Sends (game_samples, result, moves_made) to result_queue
    """
    # Initialize environment and agent for this worker
    env = ChessEnv(demo_mode=False)
    state_size = (8, 8, 12)
    agent = MCTSAgent(state_size=state_size, n_simulations=n_simulations)
    
    # Wait for initial weights from main process
    try:
        weights = weight_queue.get(timeout=10)
        if weights is None:  # Poison pill to terminate
            return
        agent.set_weights(weights)
    except:
        print(f"Worker {worker_id}: Failed to get initial weights")
        return
    
    # Play one game
    env.reset(fen=start_fen, difficulty=difficulty if not start_fen else None)
    moves_made = 0
    game_samples = []  # list of (state, improved_policy, player_color)
    
    while not env.done and moves_made < max_moves:
        state = env.get_state()
        current_player = env.board.turn
        
        # Run MCTS search
        move, improved_policy, v_pi = agent.simulate(env)
        
        if move is None:
            break
        
        # Store sample
        game_samples.append((state, improved_policy, current_player))
        
        # Play move
        _, reward, done = env.step(move)
        moves_made += 1
    
    # Get game result
    result = env.board.result(claim_draw=True)
    
    # Send results back to main process
    result_queue.put((worker_id, game_samples, result, moves_made))


class SelfPlayWorkerPool:
    """Manages a pool of worker processes for parallel self-play."""
    
    def __init__(self, num_workers, n_simulations, start_fen=None, difficulty=0, max_moves=50):
        self.num_workers = num_workers
        self.n_simulations = n_simulations
        self.start_fen = start_fen
        self.difficulty = difficulty
        self.max_moves = max_moves
        
        # Queues for communication
        self.weight_queues = [mp.Queue() for _ in range(num_workers)]
        self.result_queue = mp.Queue()
        
        # Worker processes
        self.processes = []
        
    def start_workers(self):
        """Start all worker processes."""
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop,
                args=(i, self.weight_queues[i], self.result_queue)
            )
            p.start()
            self.processes.append(p)
    
    def _worker_loop(self, worker_id, weight_queue, result_queue):
        """Continuous loop for a worker to play multiple games."""
        env = ChessEnv(demo_mode=False)
        state_size = (8, 8, 12)
        agent = MCTSAgent(state_size=state_size, n_simulations=self.n_simulations)
        
        while True:
            # Wait for weights
            try:
                weights = weight_queue.get(timeout=1)
                if weights is None:  # Poison pill
                    break
                agent.set_weights(weights)
            except:
                continue
            
            # Play one game
            env.reset(fen=self.start_fen, difficulty=self.difficulty if not self.start_fen else None)
            moves_made = 0
            game_samples = []
            
            while not env.done and moves_made < self.max_moves:
                state = env.get_state()
                current_player = env.board.turn
                
                move, improved_policy, v_pi = agent.simulate(env)
                if move is None:
                    break
                
                game_samples.append((state, improved_policy, current_player))
                _, reward, done = env.step(move)
                moves_made += 1
            
            result = env.board.result(claim_draw=True)
            result_queue.put((worker_id, game_samples, result, moves_made))
    
    def broadcast_weights(self, weights):
        """Send updated weights to all workers."""
        for queue in self.weight_queues:
            queue.put(weights)
    
    def collect_games(self, num_games):
        """Collect completed games from workers."""
        games = []
        for _ in range(num_games):
            try:
                game_data = self.result_queue.get(timeout=300)  # 5 min timeout
                games.append(game_data)
            except:
                print(f"Warning: Timeout waiting for game results")
                break
        return games
    
    def terminate(self):
        """Gracefully terminate all workers."""
        # Send poison pills
        for queue in self.weight_queues:
            queue.put(None)
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
