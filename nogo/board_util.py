"""
board_util.py
Utility functions for Go board.
"""

import numpy as np

"""
Encoding of colors on and off a Go board.
FLODDFILL is used internally for a temporary marker
"""
EMPTY = 0
BLACK = 1
WHITE = 2
BORDER = 3

def is_black_white(color):
    return color == BLACK or color == WHITE
"""
Encoding of special pass move
"""
PASS = None

"""
Encoding of "not a real point", used as a marker
"""
NULLPOINT = 0

"""
The largest board we allow. 
To support larger boards the coordinate printing needs to be changed.
"""
MAXSIZE = 25

"""
where1d: Helper function for using np.where with 1-d arrays.
The result of np.where is a tuple which contains the indices 
of elements that fulfill the condition.
For 1-d arrays, this is a singleton tuple.
The [0] indexing is needed toextract the result from the singleton tuple.
"""
def where1d(condition):
    return np.where(condition)[0]

def coord_to_point(row, col, boardsize):
    """
    Transform two dimensional (row, col) representation to array index.

    Arguments
    ---------
    row, col: int
             coordinates of the point  1 <= row, col <= size

    Returns
    -------
    point
    
    Map (row, col) coordinates to array index
    Below is an example of numbering points on a 3x3 board.
    Spaces are added for illustration to separate board points 
    from BORDER points.
    There is a one point BORDER between consecutive rows (e.g. point 12).
    
    16   17 18 19   20

    12   13 14 15
    08   09 10 11
    04   05 06 07

    00   01 02 03

    File board_util.py defines the mapping of colors to integers,
    such as EMPTY = 0, BORDER = 3.
    For example, the empty 3x3 board is encoded like this:

    3  3  3  3  3
    3  0  0  0
    3  0  0  0
    3  0  0  0
    3  3  3  3

    This board is represented by the array
    [3,3,3,3,  3,0,0,0,  3,0,0,0,  3,0,0,0,  3,3,3,3,3]
    """
    assert 1 <= row
    assert row <= boardsize
    assert 1 <= col
    assert col <= boardsize
    NS = boardsize + 1
    return NS * row + col

class GoBoardUtil(object):
    
    @staticmethod
    def generate_legal_moves(board, color):
        """
        generate a list of all legal moves on the board.
        Does not include the Pass move.

        Arguments
        ---------
        board : np.array
            a SIZExSIZE array representing the board
        color : {'b','w'}
            the color to generate the move for.
        """
        moves = board.get_empty_points()
        legal_moves = []
        for move in moves:
            if board.is_legal(move, color):
                legal_moves.append(move)
        return legal_moves

    @staticmethod       
    def generate_random_move(board, color, use_eye_filter):
        """
        Generate a random move.
        Return PASS if no move found

        Arguments
        ---------
        board : np.array
            a 1-d array representing the board
        color : BLACK, WHITE
            the color to generate the move for.
        """
        moves = board.get_empty_points()
        np.random.shuffle(moves)
        for move in moves:
            legal = not (use_eye_filter and board.is_eye(move, color)) \
                    and board.is_legal(move, color)
            if legal:
                return move
        return PASS

    @staticmethod
    def opponent(color):
        return WHITE + BLACK - color    

    @staticmethod
    def get_twoD_board(goboard):
        """
        Return: numpy array
        a two dimensional numpy array with the stones as the goboard.
        Does not pad with BORDER
        Rows 1..size of goboard are copied into rows 0..size - 1 of board2d
        """
        size = goboard.size
        board2d = np.zeros((size, size), dtype = np.int32)
        for row in range(size):
            start = goboard.row_start(row + 1)
            board2d[row, :] = goboard.board[start : start + size]
        return board2d
