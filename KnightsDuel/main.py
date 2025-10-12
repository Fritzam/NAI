from easyAI import TwoPlayerGame, AI_Player, Negamax

class KnightDuel(TwoPlayerGame):
    """
    5x5 board. Two knights (X and O) start in opposite corners.
    Knights move like in chess. Visited squares become blocked (#).
    The player who cannot move loses.
    """

    KNIGHT_MOVES = [(-2, -1), (-1, -2), (1, -2), (2, -1),
                    (2, 1), (1, 2), (-1, 2), (-2, 1)]

    def __init__(self, players):
        self.players = players
        self.size = 5
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.knight_positions = {1: (0, 0), 2: (4, 4)}  # start positions
        self.board[0][0] = 1
        self.board[4][4] = 2
        self.current_player = 1

    # ----------------------------------------------
    def possible_moves(self):
        x, y = self.knight_positions[self.current_player]
        moves = []
        for dx, dy in self.KNIGHT_MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == 0:
                moves.append(f"{nx},{ny}")
        return moves

    # ----------------------------------------------
    def make_move(self, move):
        nx, ny = map(int, move.split(','))
        x, y = self.knight_positions[self.current_player]
        self.board[x][y] = -1  # block the old square
        self.board[nx][ny] = self.current_player
        self.knight_positions[self.current_player] = (nx, ny)

    # ----------------------------------------------
    def lose(self):
        return self.possible_moves() == []

    def is_over(self):
        return self.lose()

    def scoring(self):
        # negative if losing
        return -100 if self.lose() else 0

    # ----------------------------------------------
    def show(self):
        symbols = {0: ".", -1: "#", 1: "X", 2: "O"}
        print("\n".join(" ".join(symbols[self.board[x][y]] for y in range(self.size)) for x in range(self.size)))
        print()

# ----------------------------------------------
if __name__ == "__main__":
    # both AI use Negamax with depth 4
    ai_algo_1 = Negamax(4)
    ai_algo_2 = Negamax(4)

    game = KnightDuel([AI_Player(ai_algo_1), AI_Player(ai_algo_2)])
    game.play()

    print(f"Game over! Winner: Player {game.opponent_index}")