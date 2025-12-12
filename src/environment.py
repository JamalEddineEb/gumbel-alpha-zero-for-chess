import chess
import numpy as np
import random

class ChessEnv:
    def __init__(self, demo_mode=False):
        self.board = chess.Board()
        self.mates = 0
        self.steps = 0
        self.demo_mode = demo_mode
        self.renderer = None
        self.reset()

    def reset(self, random_moves=0, fen=None, difficulty=None):
        """
        Reset the environment.
        Args:
            random_moves: Number of random moves to apply (for Hard/Random mode).
            fen: Specific FEN to load.
            difficulty: 0=Easy (Mate-in-few), 1=Medium (Close pieces), 2=Hard (Random).
                        If None, defaults to Hard (standard random generation).
        """
        self.board.reset()
        
        if fen:
            self.board.set_fen(fen)
        elif difficulty is not None:
            if difficulty == 0:
                self._generate_easy_position()
            elif difficulty == 1:
                self._generate_medium_position()
            else:
                self._generate_random_position()
        elif random_moves > 0:
            # Legacy support or explicit random moves
            for _ in range(random_moves):
                if self.board.is_game_over():
                    break
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    self.board.push(random.choice(legal_moves))
        else:
             # Default to random position if nothing specified
             self._generate_random_position()

        self.steps = 0
        self.done = False
        return self.get_state()

    def _generate_easy_position(self):
        self.board.set_fen("5K1k/8/8/8/8/3Q4/8/8 w - - 0 1")

    def _generate_medium_position(self):
        self.board.set_fen("7k/8/5K2/8/8/3Q4/8/8 w - - 0 1")
    def _generate_random_position(self):
        self.board.clear()
        
        # Place Kings
        while True:
            bk = random.randint(0, 63)
            wk = random.randint(0, 63)
            if bk == wk: continue
            
            dist = max(abs(chess.square_file(wk) - chess.square_file(bk)),
                       abs(chess.square_rank(wk) - chess.square_rank(bk)))
            if dist > 1:
                self.board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                self.board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                break
        
        # Place Queen
        while True:
            wq = random.randint(0, 63)
            if wq != bk and wq != wk:
                self.board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
                break
        
        self.board.turn = chess.WHITE
        if not self.board.is_valid() or self.board.is_checkmate() or self.board.is_stalemate():
            # Retry if invalid
            self._generate_random_position()
    def get_state(self):
        # (8, 8, 12) representation
        # Channels 0-5: White P, N, B, R, Q, K
        # Channels 6-11: Black P, N, B, R, Q, K
        state = np.zeros((8, 8, 12), dtype=np.float32)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # 0-5 for White, 6-11 for Black
                piece_idx = piece.piece_type - 1  # 0=P, 1=N, ..., 5=K
                if piece.color == chess.BLACK:
                    piece_idx += 6
                
                rank = 7 - chess.square_rank(square)
                file = chess.square_file(square)
                state[rank, file, piece_idx] = 1.0

        return state

    def go_back(self, baseline):
        self.done = False
        while len(self.board.move_stack) > baseline:
            self.board.pop()

    def step(self, action):
        if action not in self.board.legal_moves:
            # print("wrong move : ", action)
            return self.get_state(), -1.0, True

        self.board.push(action)

        if self.demo_mode:
            self.renderer.render_board(self.board)
        
        reward = 0.0
        self.done = False

        if self.board.is_checkmate():
            reward = 1.0
            self.mates += 1
            self.done = True

        elif self.board.is_stalemate() or self.board.is_game_over() or self.board.is_insufficient_material():
            reward = 0.0
            self.done = True
        elif self.board.can_claim_draw():
            # 50-move rule or 3-fold repetition
            reward = 0.0
            self.done = True
        
        return self.get_state(), reward, self.done

    def get_legal_actions(self):
        return list(self.board.legal_moves)
    
    def get_fen(self):
        return self.board.fen()

