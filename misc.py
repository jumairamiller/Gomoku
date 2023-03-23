import sys
import numpy as np
'''
This class hold helper methods for determining 
results for each players move
-------------------------------------------------------------------------------------
'''

"""
Looks at the board determines move is legal if 
- location we want to place a player on (player = moveLoc[i]) is within the board 
- location is empty
"""
def legalMove(board, moveLoc):
    BOARD_SIZE = board.shape[0]
    # if boundaries of board are violated, return false (i.e.  illegal move)
    # NOTE: 0 and 1 are x and y coordinates of location
    if moveLoc[0] < 0 or moveLoc[0] >= BOARD_SIZE or \
       moveLoc[1] < 0 or moveLoc[1] >= BOARD_SIZE:
        return False
    # if the space within the bound is empty, return true (legal move)
    if board[moveLoc] == 0:
        return True
    # otherwise (if within bounds and not empty), return false
    return False

"""

"""
def rowTest(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    mask = np.ones(X_IN_A_LINE, dtype=int)*playerID

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE-X_IN_A_LINE+1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r,c+i] != playerID:
                    flag = False
                    break
            if flag:
                return True

    return False        

def diagTest(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    for r in range(BOARD_SIZE - X_IN_A_LINE + 1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r+i,c+i] != playerID:
                    flag = False
                    break
            if flag:
                return True
    return False

def winningTest(playerID, board, X_IN_A_LINE):  
    if rowTest(playerID, board, X_IN_A_LINE) or diagTest(playerID, board, X_IN_A_LINE):
        return True

    boardPrime = np.rot90(board)
    if rowTest(playerID, boardPrime, X_IN_A_LINE) or diagTest(playerID, boardPrime, X_IN_A_LINE):
        return True
    
    return False
