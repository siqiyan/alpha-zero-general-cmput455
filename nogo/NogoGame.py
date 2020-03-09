from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
#from .NogoLogic import Board
from .simple_board import SimpleGoBoard as Board
from .board_util import GoBoardUtil, BLACK, WHITE, coord_to_point

import numpy as np

class NogoGame(Game):
    square_content = {
        -1: "w",
        +0: "-",
        +1: "b"
    }


    @staticmethod
    def getSquarePiece(piece):
        return NogoGame.square_content[piece]


    def __init__(self, n):
        self.n = n
        self.board = Board(n)
        self.root = None

    def beginSearch(self):
        self.root = self.board.copy()
        self.board, self.root = self.board, self.root


    def inSearch(self):
        self.board = self.root.copy()


    def endSearch(self):
        self.board = self.root.copy()
        self.root = None

    def copy(self):
        copy = NogoGame(self.n)
        copy.board = self.board.copy()

    def getInitBoard(self):
        # return initial board (numpy board)
        self.board.reset(self.n)
        return self.get_pieces()

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board , player, action, fast=False):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if fast:
            move = (int(action/self.n), action%self.n)
            board[move[0]][move[1]] = player
            return board, -player
        action = self.convert_point(action)
        self.board.play_move(action, self.board.current_player)
        return self.get_pieces(), -self.convert_back_color()


    def convert_color(self, player):
        if player == 1:
            return BLACK
        elif player == -1:
            return WHITE
        else: return None

    def convert_back_color(self):
        if self.board.current_player == WHITE:
            return -1
        if self.board.current_player == BLACK:
            return 1
        else: return None

    def convert_coord(self, coord):
        return ((coord[0] + 1), (coord[1] + 1))

    def convert_point(self, point):
        coord = (int(point/self.n), point%self.n)
        coord = self.convert_coord(coord)
        return coord_to_point(coord[0],coord[1],self.n)

    def convert_back_point(self, point):
        row, col = self.board._point_to_coord(point)
        return (row - 1)*(self.n) + (col - 1)

    def getValidMoves(self, _, __):
        # return a fixed size binary vector
        color = self.board.current_player
        moves = self.board.get_empty_points()
        legal = []
        for move in moves:
            if self.board.is_legal(move, color):
                legal.append(move)
        valids = [0]*self.getActionSize()
        for point in legal:
            valids[self.convert_back_point(point)] = 1
        return np.array(valids)


    def getGameEnded(self, _, __):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        legal_moves = self.getValidMoves(_,__)
        if legal_moves.max() == 1:
            return 0
        else:
            return -self.convert_back_color()

    def getCanonicalForm(self, _, __):
        # return state if player==1, else return -state if player==-1
        return self.convert_back_color()*self.get_pieces()

    def get_pieces(self):
        pieces = GoBoardUtil.get_twoD_board(self.board)
        empty = (pieces == 0).astype(int)
        empty = np.multiply(-3,empty)
        pieces = np.multiply(-2,pieces)
        pieces = np.add(3, pieces)
        pieces = np.add(empty, pieces)
        return pieces

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(NogoGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
