"""
Pojedynek skoczków na planszy 5x5 z użyciem easyAI.


Opis skrótowy:
- Dwóch graczy AI ( figury X i O) zaczyna w przeciwległych rogach.
- Ruchy jak w szachach.
- Odwiedzone pola stają się zablokowane (#) i nie można na nie wrócić.
- Przegrywa ten, kto nie ma już dozwolonego ruchu.
"""

from easyAI import TwoPlayerGame, AI_Player, Negamax


class KnightDuel(TwoPlayerGame):

    """
    Klasa obsługująca grę, dziedzicząca z TwoPlayerGame z modułu easyAI.
    Dwóch graczy porusza się jak skoczki w szachach - gracz, który nie ma już wolnego pola do skoku przegrywa.
        
    """

    """
    Dozwolone ruchy dla każdego gracza, wartości są stałe.

    """
    KNIGHT_MOVES = [(-2, -1), (-1, -2), (1, -2), (2, -1),
                    (2, 1), (1, 2), (-1, 2), (-2, 1)]

    def __init__(self, players):
        """
        Inicjalizacja gry.
        Players to lista graczy wymagana przez easyAI.
        Size - rozmiar planszy określony jako 5x5
        Board - lista w której jest konstruowana plansza.
        Knights_positions - początkowe pozycje skoczków.
        Current_player - pierwsyz gracz zaczyna.

        """
        self.players = players
        self.size = 5
        self.board = []
        for value in range(self.size):
            wiersz = [0] * self.size
            self.board.append(wiersz)
        self.knight_positions = {1: (0, 0), 2: (4, 4)}
        self.board[0][0] = 1
        self.board[4][4] = 2
        self.current_player = 1

    def possible_moves(self):
        """
        Funkcja określa i zwraca listę wszystkich możliwych ruchów dla aktualnego gracza.

        """
        x, y = self.knight_positions[self.current_player]
        possible_moves = []


        """
        Sprawdź przesunięcie - jeśli ruch nie wyprowadzi Cię poza granicę mapy/na zajęte pole to dodaj go do
        listy zezwolonych posunięć i zwróć listę.    

        """
        for dx, dy in self.KNIGHT_MOVES:
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue

            if self.board[nx][ny] == 0:
                move = f"{nx},{ny}"
                possible_moves.append(move)


        return possible_moves


    def make_move(self, move):
        """
        Wykonaj dozwolony ruch i zablokuj pole na którym uprzednio stał skoczek.

        """

        nx, ny = map(int, move.split(','))
        x, y = self.knight_positions[self.current_player]
        self.board[x][y] = -1
        self.board[nx][ny] = self.current_player
        self.knight_positions[self.current_player] = (nx, ny)

    def lose(self):
        """
        Jeżeli wyczerpiesz listę możliwych ruchów zwróc ją jako pustą.

        """
        return self.possible_moves() == []

    def is_over(self):
        """
        Metoda wymagana przez easyAI.
        Wywołuje funkcję lose()

        """
        return self.lose()

    def scoring(self):
        """
        Jeżeli została wywołana funkcja lose ustaw wynik gracza na -100. Jeśli nie, to na 0.

        """
        if self.lose():
            return -100
        return 0

    def show(self) -> None:
        """
        Rysuje planszę w formie tekstowej.

        Legenda:
        . puste pole
        # zablokowane pole
        X skoczek bota 1
        O skoczek bota 2

        """
        symbols = {0: ".", -1: "#", 1: "X", 2: "O"}

        for x in range(self.size):
            row = []
            for y in range(self.size):
                symbol = symbols[self.board[x][y]]
                row.append(symbol)
            print(" ".join(row))
        print()

if __name__ == "__main__":

    """
    Algorytmy dla botów - obydwa wykorzystują Negamax o głębi 4.

    """
    ai_player_1_algorithm = Negamax(4)
    ai_player_2_algorithm = Negamax(4)

    game = KnightDuel([AI_Player(ai_player_1_algorithm), AI_Player(ai_player_2_algorithm)])
    game.play()

    print(f"Game over! Winner: Player {game.opponent_index}")