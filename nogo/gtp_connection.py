"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import signal, os
import traceback
from sys import stdin, stdout, stderr
from board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, PASS, \
                       MAXSIZE, coord_to_point
import numpy as np
import re
import time
import random

class GtpConnection():

    def __init__(self, go_engine, board, debug_mode = False):
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self.totalTime = 0
        self.count = 0
        self.nodeExp = 0
        self.timeLimit = 1
        self.to_play = BLACK
        #H table is a dictionary that stores (state,value) pairs
        #value  =  Black win -> 1, White win -1
        self.H_table = {}
        
        self._winner = ''
        self._optimal_move = ''
        
        self._debug_mode = debug_mode
        self.go_engine = go_engine
        self.board = board
        self.commands = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd,
            "timelimit": self.timelimit_cmd,
            "solve":self.solve_cmd
        }

        # used for argument checking
        # values: (required number of arguments, 
        #          error message on argnum failure)
        self.argmap = {
            "boardsize": (1, 'Usage: boardsize INT'),
            "komi": (1, 'Usage: komi FLOAT'),
            "known_command": (1, 'Usage: known_command CMD_NAME'),
            "genmove": (1, 'Usage: genmove {w,b}'),
            "play": (2, 'Usage: play {b,w} MOVE'),
            "legal_moves": (1, 'Usage: legal_moves {w,b}'),
            "timelimit": (1, 'Usage: timelimit INT, 1 <= INT <= 100'),
        }
    
    def write(self, data):
        stdout.write(data) 

    def flush(self):
        stdout.flush()

    def start_connection(self):
        """
        Start a GTP connection. 
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command):
        """
        Parse command string and execute it
        """
        if len(command.strip(' \r\t')) == 0:
            return
        if command[0] == '#':
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements = command.split()
        if not elements:
            return
        command_name = elements[0]; args = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".
                               format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error('Unknown command')
            stdout.flush()

    def has_arg_error(self, cmd, argnum):
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg):
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg):
        """ Send error msg to stdout """
        stdout.write('? {}\n\n'.format(error_msg))
        stdout.flush()

    def respond(self, response=''):
        """ Send response to stdout """
        stdout.write('= {}\n\n'.format(response))
        stdout.flush()

    def reset(self, size):
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self):
        return str(GoBoardUtil.get_twoD_board(self.board))
        
    def protocol_version_cmd(self, args):
        """ Return the GTP protocol version being used (always 2) """
        self.respond('2')

    def quit_cmd(self, args):
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args):
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args):
        """ Return the version of the  Go engine """
        self.respond(self.go_engine.version)

    def clear_board_cmd(self, args):
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args):
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()
     
    #newly added   
    def timelimit_cmd(self, args):
        """
        Reset the game with new timelimit args[0]
        """
        self.timeLimit = int(args[0])
        self.respond()        

    def showboard_cmd(self, args):
        self.respond('\n' + self.board2d())

    def komi_cmd(self, args):
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args):
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args):
        """ list all supported GTP commands """
        self.respond(' '.join(list(self.commands.keys())))

    def legal_moves_cmd(self, args):
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        moves = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def play_cmd(self, args):
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        try:
            board_color = args[0].lower()
            board_move = args[1]
            if board_color != "b" and board_color !="w":
                self.respond("illegal move: \"{}\" wrong color".format(board_color))
                return
            color = color_to_int(board_color)
            #change turn to the other player
            self.to_play = GoBoardUtil.opponent(color)
            if args[1].lower() == 'pass':
                self.respond("illegal move: \"{} {}\" wrong coordinate".format(args[0], args[1]))
                return
            coord = move_to_coord(args[1], self.board.size)
            if coord:
                move = coord_to_point(coord[0],coord[1], self.board.size)
            else:
                self.error("Error executing move {} converted from {}"
                           .format(move, args[1]))
                return
            if not self.board.play_move(move, color):
                self.respond("illegal move: \"{} {}\" ".format(args[0], board_move))
                return
            else:
                self.debug_msg("Move: {}\nBoard:\n{}\n".
                                format(board_move, self.board2d()))
            self.respond()
        except Exception as e:
            self.respond('illegal move: \"{} {}\" {}'.format(args[0], args[1], str(e)))

    def solve_helper(self):

        
        winner = 'unknown'
        
        #the copy of board can be viewed as a state
        cp_board = self.board.copy()
        
        start = time.time()
               
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeLimit)
        try:
            value,move = self.advanced_search(cp_board,81,-1,1)
        except Exception as e:
            value,move = 0,None
        #print("nodeExp",self.nodeExp)
        #print("count",self.count)
        
        signal.alarm(0) 
        
        end = time.time()
        print("time: ",end - start) 
        
        #print("partial time: ",self.totalTime) 
        if value == 1:
            winner = 'b'
        elif value == -1:
            winner = 'w'
        
        
        
        if (winner == 'b' and self.to_play !=BLACK) or (winner == 'w' and self.to_play !=WHITE):
            move = None

        return winner,move
    
    #newly added    
    def solve_cmd(self,args):        
        moveStr = ''
        winner,move = self.solve_helper()
        if move:
            moveStr = ' '+ coord_to_move(move,self.board.size)            
        self.respond(winner+moveStr)

   #alpha beta pruning, referencing from wikipedia: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
   #color is the player. black is max player, white is min player
    def ab_search(self, color, copy_of_board, depth, alpha, beta):
        _alpha = alpha
        _beta = beta
        bestMove = None
        #base case, no more legal move
        #print(GoBoardUtil.generate_legal_moves(copy_of_board, color))
        if depth == 0 or (GoBoardUtil.generate_legal_moves(copy_of_board, color) == []):
            #depth should always be >0
            
            #since NOGO cannot capture nor suiside, if last move is by WHITE/BLACK, it must be a BLACK/WHITE win.
            if color == WHITE:
                return 1,None
            #color == BLACK
            else:
                return -1,None
        
        #color is black; max player
        if color == BLACK:
            value = -1000000
            #make a copy of current state
            
            allmoves = GoBoardUtil.generate_legal_moves(copy_of_board, color)
            #print("allmoves:")
            #print(allmoves)

            for move in allmoves:
                child = copy_of_board.copy()
                child.play_move(move, color)

                childValue,_ = self.ab_search(WHITE,child,depth-1,_alpha,_beta)
                value = max(value,childValue)
                _alpha = max(_alpha,value)
                bestMove = move
                #beta cut-off
                if _alpha >= _beta:
                    break
            return value,bestMove
        #color is white; min player
        else:
            value = 1000000
            allmoves = GoBoardUtil.generate_legal_moves(copy_of_board, color)
            #print("allmoves:")
            #print(allmoves)
            for move in allmoves:
                child = copy_of_board.copy()
                child.play_move(move, color)
                childValue,_ = self.ab_search(BLACK,child,depth-1,_alpha,_beta)
                value = min(value,childValue)
                _beta = min(_beta,value)
                bestMove = move
                #alpha cut-off
                if _alpha >= _beta:
                    break
            return value,bestMove           
        
    def advanced_search(self,copy_of_board,depth,alpha,beta):
        
        _alpha = alpha
        _beta = beta
        bestMove = None
        self.nodeExp += 1
        
        #base case, depth 0
        if depth == 0:
            return 0,None
        
        #Start = time.time()
        allmoves = GoBoardUtil.generate_legal_moves(copy_of_board, copy_of_board.current_player)
        #End =time.time()
        #self.totalTime += End-Start 
        
        #base case, no more legal move
        if allmoves == []:
    
            #since NOGO cannot capture nor suiside, if last move is by WHITE/BLACK, it must be a BLACK/WHITE win.
            if copy_of_board.current_player == WHITE:
                self.H_table[self.tuple_to_str(self.matrix_to_tuple(GoBoardUtil.get_twoD_board(copy_of_board),copy_of_board.size))] = 1
                return 1,None
            #color == BLACK
            else:
                self.H_table[self.tuple_to_str(self.matrix_to_tuple(GoBoardUtil.get_twoD_board(copy_of_board),copy_of_board.size))] = -1
                return -1,None
    
        
        
        searchedMoves = []
        unsearchedMoves = []
        unsearched = {}
        searchedValue = {}
        
        isoSet = set()
        singleMoveIsoSet = set()
        for move in allmoves:
            singleMoveIsoSet.clear()
            child = copy_of_board.copy()
            child.play_move(move, copy_of_board.current_player)

            #get all isomorphics of the board, in order to prunning as many as redundent states possible
            isomorphics = self.get_all_isomorphic(GoBoardUtil.get_twoD_board(child),child.size)
       
            found = False

            for iso in isomorphics:
                if self.tuple_to_str(iso) in self.H_table:
                    found = True
                    searchedMoves.append(move)
                    searchedValue[move] = self.H_table[self.tuple_to_str(iso)]
                    break
                if iso in isoSet:
                    found = True
                    break
                else:
                    isoSet.add(iso) 
                    singleMoveIsoSet.add(iso)
                 
            if not found:
                '''
                the following is the heuristic I created for ordering the moves:
                (1) eye-filling is the last thing we want to do;
                (2) the few the number of player's stones with MD 1, the better;
                (3) the more the number of opponent's stones with MD 1, the better;
                (4) the more the number of player's stones with MD 2, the better;
                '''

                num_same = 49
                dis1 = [move+1,move-1,move+child.size+1,move-child.size-1]
                dis2 = [move+2,move-2,move+2*(child.size+1),move-2*(child.size+1),move+child.size+2,move-child.size-2,move+child.size,move-child.size]
                
                valid1 = []
                
                for point in dis1:
                    x = point%(child.size+1)
                    y = point//(child.size+1) 
                    if 1<=x<=child.size and 1<=y<=child.size:
                        valid1.append(point)

                valid2 = []
                for point in dis2:
                    x = point%(child.size+1)
                    y = point//(child.size+1) 
                    if 1<=x<=child.size and 1<=y<=child.size:
                        valid2.append(point) 
                
                if copy_of_board.is_eye(move,copy_of_board.current_player):
                    num_same += 1000
                for point in valid1:
                    if child.get_color(point)==copy_of_board.current_player:
                        num_same += 100
                    if child.get_color(point)== BLACK+WHITE-copy_of_board.current_player:
                        num_same -= 10


                for point in valid2:
                    if child.get_color(point)==copy_of_board.current_player:
                        num_same -= 1

                unsearched[move] =  num_same
 
                
        #print("dic:",unsearched)
        #print("searched:",searchedMoves)
        
        #sorting unsearched moves by the heuristic value
        sorted_x = sorted(unsearched.items(), key=lambda kv: kv[1])
        for item in sorted_x:
            unsearchedMoves.append(item[0])  
        
        orderedMoves = searchedMoves + unsearchedMoves

        self.count += len(allmoves) - len(orderedMoves)
        
        
        state = self.tuple_to_str(self.matrix_to_tuple(GoBoardUtil.get_twoD_board(copy_of_board),copy_of_board.size))

        #below is normal alpha-beta search
        #color is black; max player
        if copy_of_board.current_player == BLACK:
            value = -1000000
            #make a copy of current state
            
            for move in orderedMoves:
                if move  in searchedMoves:
                    childValue = searchedValue[move]
                else:
                    child = copy_of_board.copy()
                    child.play_move(move, copy_of_board.current_player)                    
                    childValue,_ = self.advanced_search(child,depth-1,_alpha,_beta)
                    #childValue,_ = self.advanced_search(copy_of_board,depth-1,_alpha,_beta)
                value = max(value,childValue)
                _alpha = max(_alpha,value)
                bestMove = move
                #beta cut-off
                if _alpha >= _beta:
                    break
            self.H_table[state] = value  
            return value,bestMove
        #color is white; min player
        else:
            value = 1000000

            for move in orderedMoves:
                if move  in searchedMoves:
                    childValue = searchedValue[move]
                else:
                    child = copy_of_board.copy()
                    child.play_move(move, copy_of_board.current_player)                    
                    #childValue,_ = self.advanced_search(copy_of_board,depth-1,_alpha,_beta)
                    childValue,_ = self.advanced_search(child,depth-1,_alpha,_beta)
                value = min(value,childValue)
                _beta = min(_beta,value)
                bestMove = move
                #alpha cut-off
                if _alpha >= _beta:
                    break
            self.H_table[state] = value  
            return value,bestMove             


    
        
    def get_all_isomorphic(self, board_2d,size):
        """
        input: matrix of a board
        output: a set of tuples
        """      
        isomorphics = set()
        
        
        #original

        #print("mat to tuple:")
        #print(self.matrix_to_tuple(board_2d,size))
        isomorphics.add(self.matrix_to_tuple(board_2d,size))
        
        #return isomorphics
        tmp_board = []
        #reflectional sym, 2 cases
        
        #swap rows
        cp_board_2dx = board_2d.copy()
        for i in range(size//2):
            tmp = cp_board_2dx[i,:].copy()
            cp_board_2dx[i,:] = cp_board_2dx[size-1-i,:] 
            cp_board_2dx[size-1-i,:]=tmp
 
        isomorphics.add(self.matrix_to_tuple(cp_board_2dx,size)) 
        
        #swap columns
        cp_board_2dy = board_2d.copy()
        for j in range(size//2):
            for i in range(size):
                tmp = cp_board_2dy[i,j]
                cp_board_2dy[i,j] = cp_board_2dy[i,size-1-j] 
                cp_board_2dy[i,size-1-j] = tmp
   
        isomorphics.add(self.matrix_to_tuple(cp_board_2dy,size))         
        
        #rotational sym, 3 cases
        board_90 = np.rot90(board_2d)
       #board_90 = self.rotateMatrix(board_2d,size)
        isomorphics.add(self.matrix_to_tuple(board_90,size)) 
        
        #reflectional sym of 90 degree, 2 cases
        #swap rows
        cp_board_90x = board_90.copy()
        for i in range(size//2):
            tmp = cp_board_90x[i,:].copy()
            cp_board_90x[i,:] = cp_board_90x[size-1-i,:] 
            cp_board_90x[size-1-i,:] = tmp
 
        isomorphics.add(self.matrix_to_tuple(cp_board_90x,size)) 
        
        #swap columns
        cp_board_90y = board_90.copy()
        for j in range(size//2):
            for i in range(size):
                tmp = cp_board_90y[i,j]
                cp_board_90y[i,j] = cp_board_90y[i,size-1-j] 
                cp_board_90y[i,size-1-j] = tmp
   
        isomorphics.add(self.matrix_to_tuple(cp_board_90y,size))            
        
        #print("90",board_90)
        board_180 = np.rot90(board_90)
        #print("180",board_180)
        isomorphics.add(self.matrix_to_tuple(board_180,size))  

        board_270 = np.rot90(board_180)
        #print("270",board_270)
        isomorphics.add(self.matrix_to_tuple(board_270,size))         
        #board_180 = self.rotateMatrix(board_90,size)
        #isomorphics.add(self.matrix_to_tuple(board_180,size)) 
        #board_270 = self.rotateMatrix(board_180,size)
        #isomorphics.add(self.matrix_to_tuple(board_270,size))         
        
        return isomorphics
    
    
    
    def matrix_to_tuple(self,matrix,dim):      
        board1d = np.zeros((dim* dim), dtype = np.int32)
        for i in range(dim):
            board1d[i*dim:i*dim+dim] = matrix[i,:]
        return tuple(board1d)    

    def get_oneD_board(self,goboard):
        """
        Return: numpy array
        a 1-d numpy array with the stones as the goboard.
        Does not pad with BORDER
        Rows 1..size of goboard are copied into rows 0..size - 1 of board2d
        """
        size = goboard.size
        board1d = np.zeros((size* size), dtype = np.int32)
        for row in range(size):
            start = goboard.row_start(row + 1)
            board1d[row*size:row*size+size] = goboard.board[start : start + size]
        return board1d    
    
    def tuple_to_str(self,tup):
        res = ''
        for i in tup:
            res +=  str(int(i))
        return res        
    
    #genemove overrided
    def genmove_cmd(self, args):
        """
        Generate a move for the color args[0] in {'b', 'w'}, for the game of gomoku.
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        
        self.to_play = color
        winnerStr,optMove = self.solve_helper()
        winner = EMPTY
        if winnerStr=='b':
            winner = BLACK
        elif winnerStr =='w':
            winner = WHITE
        #if current player is winner, we will take bestmove; otherwise we should take a random move
        if board_color == winner:
            move = optMove
        else:
            move = GoBoardUtil.generate_random_move(self.board, color,False)
        move_coord = point_to_coord(move, self.board.size)
        move_as_string = format_point(move_coord)
        if self.board.is_legal(move, color):
            self.board.play_move(move, color)
            self.respond(move_as_string)
        else:
            self.respond("resign")


    def gogui_rules_game_id_cmd(self, args):
        self.respond("NoGo")
    
    def gogui_rules_board_size_cmd(self, args):
        self.respond(str(self.board.size))
    
    def legal_moves_cmd(self, args):
        """
            List legal moves for color args[0] in {'b','w'}
            """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        moves = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def gogui_rules_legal_moves_cmd(self, args):
        empties = self.board.get_empty_points()
        color = self.board.current_player
        legal_moves = []
        for move in empties:
            if self.board.is_legal(move, color):
                legal_moves.append(move)

        gtp_moves = []
        for move in legal_moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)
    
    def gogui_rules_side_to_move_cmd(self, args):
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)
    
    def gogui_rules_board_cmd(self, args):
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)
    
    def gogui_rules_final_result_cmd(self, args):
        empties = self.board.get_empty_points()
        color = self.board.current_player
        legal_moves = []
        for move in empties:
            if self.board.is_legal(move, color):
                legal_moves.append(move)
        if not legal_moves:
            result = "black" if self.board.current_player == WHITE else "white"
        else:
            result = "unknown"
        self.respond(result)

    def gogui_analyze_cmd(self, args):
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

def point_to_coord(point, boardsize):
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is not transformed
    """
    if point == PASS:
        return PASS
    else:
        NS = boardsize + 1
        return divmod(point, NS)

def format_point(move):
    """
    Return move coordinates as a string such as 'a1', or 'pass'.
    """
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    #column_letters = "abcdefghjklmnopqrstuvwxyz"
    if move == PASS:
        return "pass"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1]+ str(row) 
    
def move_to_coord(point_str, board_size):
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return PASS
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        # e.g. "a0"
        raise ValueError("wrong coordinate")
    if not (col <= board_size and row <= board_size):
        # e.g. "a20"
        raise ValueError("wrong coordinate")
    return row, col

def coord_to_move(move, board_size):
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    #s = point_str.lower()
    x = move%(board_size+1)
    y = move//(board_size+1)
    col = chr(x-1 + ord("a"))
    #col = col.upper()
    
    return col+str(y)


def color_to_int(c):
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK , "w": WHITE, "e": EMPTY, 
                    "BORDER": BORDER}
    return color_to_int[c] 

def handler(signum, frame):
    print('Signal handler called with signal', signum)
    raise Exception("Timeout!")
