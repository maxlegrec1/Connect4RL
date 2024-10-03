BLUE = -1
RED = 1

START_SIDE = BLUE

import numpy as np
import random

class Connect4:
    def __init__(self, to_move=START_SIDE, winner=None, pos=None):
        self.to_move = to_move
        self.winner = winner
        if pos is None:
            self.board = np.zeros((7, 7),dtype=np.float32)
        else:
            self.board = pos.copy()

        # print(self)

    def move(self, move):

        if not self.winner == None:
            print(self)
            raise ValueError(f"Player {self.winner} has already won the game")

        if not self.is_legal(move):
            raise ValueError(f"move {move} is not legal, winner : {self.winner}")

        height = self.determine_height(move)
        self.board[move, height] = self.to_move

        # update side to move and self.winner status
        self.switch_side()

        self.check_for_win()

        # print(self)
        return self

    def board_is_full(self):
        return not np.any(self.board == 0)

    def check_for_win(self):
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]  # Horizontal, Vertical, Diagonal down, Diagonal up

        for i in range(7):
            for j in range(7):
                if self.board[i, j] != 0:
                    for dx, dy in directions:
                        count = 0
                        for step in range(4):
                            x, y = i + step * dx, j + step * dy
                            if (
                                0 <= x < 7
                                and 0 <= y < 7
                                and self.board[x, y] == self.board[i, j]
                            ):
                                count += 1
                            else:
                                break
                        if count == 4:
                            self.winner = RED if self.board[i, j] == RED else BLUE
                            # print(f"Player {self.winner} won the game ! ")
                            return
        if self.board_is_full():
            self.winner = 0
            return

    def switch_side(self):
        self.to_move = -self.to_move

    def determine_height(self, move):
        height = 0
        while self.board[move, height] != 0:
            height += 1

        return height

    def is_legal(self, move):
        if move < 0 or move > 7:
            return False
        last_row = self.board.shape[1]
        return self.board[move, last_row - 1] == 0

    def legal_moves(self):

        last_row_is_zero = self.board[:, -1] == 0
        return np.arange(7)[last_row_is_zero]

    def legal_moves_mask(self):
        last_row = (self.board[:, -1]) % 2
        return -1000 * last_row
    def legal_moves_softmax(self):
        last_row = (self.board[:, -1]) % 2
        return 1 - last_row

    def __str__(self):
        # Unicode characters for the pieces and board
        blue_piece = "🔵"  # Blue circle
        red_piece = "🔴"  # Red circle
        empty = "⚪"  # White circle

        # Create the board representation
        board_str = ""
        for row in range(6, -1, -1):  # Start from the top row
            board_str += "│ "
            for col in range(7):
                if self.board[col, row] == BLUE:
                    board_str += blue_piece
                elif self.board[col, row] == RED:
                    board_str += red_piece
                else:
                    board_str += empty
                board_str += " "
            board_str += "│\n"

        # Add the bottom of the board
        # board_str += "└─" + "──" * 10 + "┘\n"

        # Add column numbers
        board_str += "   " + "  ".join(str(i) for i in range(7)) + " "

        # Add game status
        if self.winner is None:
            if self.to_move == BLUE:
                board_str += "\n\nBlue to move"
            else:
                board_str += "\n\nRed to move"
        else:
            board_str += f"\n\n{'Blue' if self.winner == BLUE else 'Red'} has won!"

        return board_str

    def copy(self):
        return Connect4(self.to_move, self.winner, self.board)

    def initialize_random(self):
        num_moves = np.random.randint(4,11)
        for i in range(num_moves + round(random.random())):
            if self.winner != None:
                break

            #sample uniformly across legal moves
            legal_moves = self.legal_moves()
            permute = np.random.permutation(len(legal_moves))
            selected_move = legal_moves[permute][0]
            self.move(selected_move)
        
