import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # raise NotImplementedError('you need to write this!')
    value = 0
    move_list = []
    move_tree = {}

    moves = [move for move in generateMoves(side, board, flags)]

    if depth == 0:
        # Base case
        return evaluate(board), [], {}

    if not side:
        # Max player
        max_val = None
        for move in moves:
            new_side, new_board, new_flags = makeMove(side, board, move[0], move[1], flags, move[2])
            cur_val, cur_list, cur_moves = minimax(new_side, new_board, new_flags, depth-1)
            if max_val is None or max_val < cur_val:
                max_val = cur_val
                if len(cur_list) == 0:
                    move_list = [move]
                else:
                    cur_list.insert(0, move)
                    move_list = cur_list
            move_tree[encode(*move)] = cur_moves
        value = max_val

    else:
        # Min player
        min_val = None
        for move in moves:
            new_side, new_board, new_flags = makeMove(side, board, move[0], move[1], flags, move[2])
            cur_val, cur_list, cur_moves = minimax(new_side, new_board, new_flags, depth - 1)
            if min_val is None or min_val > cur_val:
                min_val = cur_val
                if len(cur_list) == 0:
                    move_list = [move]
                else:
                    cur_list.insert(0, move)
                    move_list = cur_list
            move_tree[encode(*move)] = cur_moves
        value = min_val

    return value, move_list, move_tree

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # raise NotImplementedError('you need to write this!')
    value = 0
    move_list = []
    move_tree = {}

    moves = [move for move in generateMoves(side, board, flags)]

    if depth == 0:
        # Base case
        return evaluate(board), [], {}

    if not side:
        # Max player
        max_val = None
        for move in moves:
            new_side, new_board, new_flags = makeMove(side, board, move[0], move[1], flags, move[2])
            cur_val, cur_list, cur_moves = alphabeta(new_side, new_board, new_flags, depth - 1, alpha, beta)
            if max_val is None or max_val < cur_val:
                max_val = cur_val
                if len(cur_list) == 0:
                    move_list = [move]
                else:
                    cur_list.insert(0, move)
                    move_list = cur_list
            move_tree[encode(*move)] = cur_moves
            alpha = max(alpha, max_val)
            if alpha >= beta:
                # prune
                return max_val, move_list, move_tree
        value = max_val
    else:
        # Min player
        min_val = None
        for move in moves:
            new_side, new_board, new_flags = makeMove(side, board, move[0], move[1], flags, move[2])
            cur_val, cur_list, cur_moves = alphabeta(new_side, new_board, new_flags, depth - 1, alpha, beta)
            if min_val is None or min_val > cur_val:
                min_val = cur_val
                if len(cur_list) == 0:
                    move_list = [move]
                else:
                    cur_list.insert(0, move)
                    move_list = cur_list
            move_tree[encode(*move)] = cur_moves
            beta = min(beta, min_val)
            if alpha >= beta:
                # prune
                return min_val, move_list, move_tree
        value = min_val

    return value, move_list, move_tree


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    # raise NotImplementedError('you need to write this!')
    value = 0
    move_list = []
    move_tree = {}

    moves = [move for move in generateMoves(side, board, flags)]

    average_vals = []
    for move in moves:
        # Make the initial move
        new_side, new_board, new_flags = makeMove(side, board, move[0], move[1], flags, move[2])
        val_list = []
        move_dict = {}
        cur_move_list = []
        for i in range(breadth):
            # search down breadth paths
            leafval, cur_move_list, moves_tree, first_move = get_leaf_val(new_side, new_board, new_flags, depth-1, chooser)
            val_list.append(leafval)
            move_dict[encode(*first_move)] = moves_tree[encode(*first_move)]
        move_tree[encode(*move)] = move_dict
        cur_move_list.insert(0, move)
        move_list.append(cur_move_list)
        average_vals.append(sum(val_list)/len(val_list))
    if not side:
        best_move_idx = average_vals.index(max(average_vals))
    else:
        best_move_idx = average_vals.index(min(average_vals))

    value = average_vals[best_move_idx]
    move_list = move_list[best_move_idx]

    return value, move_list, move_tree


def get_leaf_val(side, board, flags, depth, chooser):
    if depth == 0:
        return evaluate(board), [], {}, None
    moves = [move for move in generateMoves(side, board, flags)]
    move = chooser(moves)
    newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
    leafval, moves_made, moves_tree, ignore = get_leaf_val(newside, newboard, newflags, depth - 1, chooser)
    if moves_made is None:
        moves_made = [move]
    else:
        moves_made.insert(0, move)
    return leafval, moves_made, {encode(*move): moves_tree}, move
