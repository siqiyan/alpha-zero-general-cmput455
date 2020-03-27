#!/usr/bin/env python3
from nogo.gtp_connection import GtpConnection, format_point, point_to_coord
from nogo.board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, coord_to_point
from nogo.simple_board import SimpleGoBoard
from MCTS import MCTS
from nogo.pytorch.NNet import NNetWrapper as nn
from nogo.NogoGame import NogoGame
import numpy as np
from utils import *

class Nogo():
    def __init__(self):
        """
        NoGo player for alpha-zero-general

        """
        self.name = "NoGoAlphaZeroGeneral"
        self.version = 1.0

        self.g = NogoGame(5)
        self.n1 = nn(self.g)
        self.n1.load_checkpoint('./pretrained_models/nogo/','temp.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        self.mcts = MCTS(self.g, self.n1, args1)
        self.n1p = lambda x: np.argmax(self.mcts.getActionProb(x, temp=0))
        
    def get_move(self, board, color):
        moves = board.get_empty_points()
        curPlayer = 1 if color == BLACK else -1
        self.g.board = board
        action = self.n1p(self.g.getCanonicalForm(board, curPlayer)) + 1
        return moves[action]

        # color = board.current_player
        # moves = board.get_empty_points()
        # legal = []
        # for move in moves:
            # if self.board.is_legal(move, color):
                # legal.append(move)
        # valids = [0]*self.getActionSize()
        # for point in legal:
            # valids[self.convert_back_point(point)] = 1
        # return np.array(valids)
    
def run():
    """
    start the gtp connection and wait for commands.
    """
    board = SimpleGoBoard(5)
    con = GtpConnection(Nogo(), board)
    con.start_connection()
    con.play_cmd()

if __name__=='__main__':
    run()
