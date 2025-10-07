class Board():
    def __init__(self):
        self.board_size = 8
        self.board = self.initialize_board()
        self.draw_pieces()

    def generate_row(self) -> list[str]:
        row = []
        for field in range(self.board_size):
            row.append(" ")
        return row

    def initialize_board(self) -> list[str]:
        board = []
        for column in range(self.board_size):
            row = self.generate_row()
            board.append(row)
        return board
    
    def fetch_board(self) -> list[str]:
        return self.board
    
    def draw_board(self) -> None:
        print("     A   B   C   D   E   F   G   H")
        for i, row in enumerate(self.board, start=1):
            print(f"{i:2} | " + " | ".join(row) + " |")

    def draw_piece(self, row, col, color):
        if color == "black":
            self.board[row][col] = "$"
        elif color == "white":
            self.board[row][col] = "*"

    def draw_pieces(self) -> None:
        for row in range(8):
            if row >= 0 and row < 3:
                for col in range(8):
                    if (row + col) % 2 == 0:
                        self.draw_piece(row, col, "black")
            if row == 3 or row == 4:
                continue
            elif row >= 5:
                for col in range(8):
                    if (row + col) % 2 == 0:
                        self.draw_piece(row, col, "white")






        