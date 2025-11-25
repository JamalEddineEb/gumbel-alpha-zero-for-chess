import chess

try:
    board = chess.Board("5K1k/8/8/8/8/3Q4/8/8 w HAha - 0 1")
    print("FEN is valid (surprisingly)")
    print(board)
    print("Legal moves:", [m.uci() for m in board.legal_moves])
    
    # Check for mate in 1 or 2
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            print(f"Mate in 1 found: {move.uci()}")
        else:
            for move2 in board.legal_moves:
                board.push(move2)
                if board.is_checkmate():
                    # This would be Black mating White, which is impossible here
                    pass
                # Check if White can mate in 2 (after Black's response)
                # This is a simple check, not a full minimax
                board.pop()
        board.pop()

except ValueError as e:
    print(f"FEN Invalid: {e}")

# Test a true Mate in 1 position
board2 = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
print("\nTest Mate in 1 Position:")
print(board2)
for move in board2.legal_moves:
    board2.push(move)
    if board2.is_checkmate():
        print(f"Mate in 1 found: {move.uci()}")
    board2.pop()
