# ♟️ Gumbel AlphaZero for Chess

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated **Gumbel AlphaZero (GAZ)** agent designed to master **Chess**. This project demonstrates the power of combining deep convolutional neural networks with **Sequential Halving MCTS** and **Gumbel noise exploration**, achieving superior sample efficiency compared to traditional AlphaZero methods.

While the default training loop focuses on the **King and Queen vs. King (KQK)** endgame for rapid demonstration, the agent and architecture are fully capable of training on **any chess position** or full games.

*The Gumbel AlphaZero agent playing against itself.*

<img src="images/demo.GIF" width="500" title="Gumbel AlphaZero Self-play">

## 🌟 Key Features

- **🚀 Gumbel-Max Exploration**: Uses Gumbel noise sampling for robust action selection without the need for high simulation counts ($N$).
- **📉 Sequential Halving**: Implements an efficient search budget allocation strategy, progressively pruning low-value branches.
- **🧠 Two-Headed Architecture**: Features a modern CNN with separate heads for **Policy** (move probabilities) and **Value** (win/loss estimation).
- **🤖 Stockfish Baseline**: Trains against the world-class **Stockfish** engine to ensure high-quality self-play data and rigorous evaluation.
- **🌐 General Purpose**: Supports loading any board state via FEN or PGN for training and inference.
- **🐳 Docker Support**: Fully containerized environment for easy reproducibility.

---

## 📂 Project Structure

```
chess-endgame-mcts/
├── docker-compose.yml      # Docker services configuration
├── dockerfile              # Docker image definition
├── images/                 # Project assets
│   └── mate.png
├── model_checkpoint.weights.h5  # Trained model weights
├── readme.md               # Project documentation
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── chess_renderer.py   # PyQt5 GUI for visualization
    ├── environment.py      # Chess environment & Stockfish wrapper
    ├── mcts_agent.py       # MCTS algorithm & Neural Network
    ├── mcts_node.py        # Tree search node definition
    ├── play.py             # Inference script (GUI/Headless)
    ├── train.py            # Training loop (Self-play)
    └── utils/              # Helper utilities
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Stockfish**: Must be installed and accessible in your system path (or configured in `src/environment.py`).

### 📦 Installation

#### Option A: Local Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chess-endgame-mcts.git
    cd chess-endgame-mcts
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

#### Option B: Docker (Recommended)

1.  Build and run the container:
    ```bash
    docker-compose up --build
    ```

---

## 💻 Usage

### 1. Training the Agent

Start the self-play training loop. By default, it generates random **KQK** positions.

```bash
python -m src.train
```

**Arguments:**
- `--fen <string>`: Start training from a specific FEN position (e.g., full starting position).
- `--pgn <path>`: Load a starting position from a PGN file.

**Example (Full Game Training):**
```bash
python -m src.train --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### 2. Playing / Demo

Watch the trained agent play in real-time.

```bash
python -m src.play
```

**Arguments:**
- `--fen <string>`: Set the starting board position.
- `--headless`: Run without the GUI (terminal output only).

**Example (Custom Endgame):**
```bash
python -m src.play --fen "8/8/3k4/8/8/3K1Q2/8/8 w - - 0 1"
```

---

## 🧠 Algorithmic Details

### The Gumbel Advantage
Traditional AlphaZero relies on PUCT (Predictor + Upper Confidence Bound applied to Trees) which requires many simulations to visit nodes. **Gumbel AlphaZero** improves this by:

1.  **Sampling**: Using Gumbel noise to select a set of candidate actions at the root.
2.  **Sequential Halving**: Allocating the simulation budget dynamically. If we have $N$ simulations, we evaluate $K$ candidates, then keep the top $K/2$, then $K/4$, and so on, doubling the visits for the survivors at each stage.

### Policy Improvement Target
The network is trained to match an improved policy $\pi'$:

$$ \pi' \propto \text{softmax}(\text{logits} + \sigma(Q_{completed})) $$

This ensures the network learns from the *search-improved* values rather than just raw visit counts.
