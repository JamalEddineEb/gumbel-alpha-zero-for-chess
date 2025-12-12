import sys
import argparse
import time

from src.mcts_agent import MCTSAgent
from src.environment import ChessEnv

# Parse arguments
parser = argparse.ArgumentParser(description='Play Chess with MCTS Agent')
parser.add_argument('--fen', type=str, help='Starting FEN string')
parser.add_argument('--headless', action='store_true', help='Run without GUI')
args = parser.parse_args()

# --- setup ---
state_size = (8, 8, 12)  # 8x8 board with 12 channels
agent = MCTSAgent(state_size, n_simulations=100)

model_file = "model_checkpoint.weights.h5"
print(f"Loading model from {model_file}")
try:
    agent.load(model_file)
except Exception as e:
    print(f"Could not load model: {e}")
    print("Starting with random weights.")

env = ChessEnv(demo_mode=False)
env.reset(fen=args.fen)

# Try to use GUI if not headless
use_gui = not args.headless
renderer = None

if use_gui:
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        from src.chess_renderer import ChessRenderer
        
        app = QApplication.instance() or QApplication(sys.argv)
        renderer = ChessRenderer(env.board)
        renderer.show()
        renderer.update_board()
        print("GUI mode enabled")
    except Exception as e:
        print(f"Could not initialize GUI: {e}")
        print("Falling back to headless mode")
        use_gui = False

MOVE_DELAY_MS = 800 

def play_step_gui():
    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    # Agent move via MCTS
    move, _, _ = agent.simulate(env)
    if move is None:
        print("No legal move; stopping.")
        return

    print(f"Move: {move}")
    env.step(move)
    renderer.update_board()

    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    QTimer.singleShot(MOVE_DELAY_MS, play_step_gui)

def play_headless():
    move_count = 0
    max_moves = 100
    
    while not env.board.is_game_over() and move_count < max_moves:
        # Agent move via MCTS
        move, _, _ = agent.simulate(env)
        if move is None:
            print("No legal move; stopping.")
            break

        print(f"Move {move_count + 1}: {move}")
        env.step(move)
        print(env.board.unicode())
        print()
        
        move_count += 1
        time.sleep(0.5)  # Small delay for readability

    print(f"Game over: {env.board.result()}")

if use_gui:
    # start loop after small delay so first frame shows
    QTimer.singleShot(MOVE_DELAY_MS, play_step_gui)
    sys.exit(app.exec_())
else:
    play_headless()
