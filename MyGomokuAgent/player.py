"""
Group 2 - Radoslav Nikolov 974054
         Jumaira Miller 983101

Strategy:
    - First move will always be either center position or the position directly above the center. This is the position
    that gives us the greatest advantage because it allows us to expand in all directions equally.
    - We always opt to be defensive and block first unless we can immediately win with our next move.
    - We block in order of increasing threat:
        a) firstly, we block closed 4s because the opponent could otherwise win with his next move. NOTE: our algorithm
            does not allow for the opponent to ever make an open 4 (i.e. 4 in a row with both ends empty)
        b) secondly, we block open 3s to prevent the opponent from making an open 4 - if the opponent was able to make
            open 4, this will be a double threat to the our player agent (i.e. no way for us to win).
        c) thirdly, we block broken 3s because this restricts the opponent from making an open 4 or open 3
            by further extending the positions which compose this broken 3.
        d) finally, we block closed three. This is the least priority as this threat can be immediately removed -
            the opponent will not be able to further extend this path of consecutive positions because it is now blocked
            at both ends.
    - If there are no immediate threats to block, search the current board for the best offensive move:
        a) we use alpha-beta pruning with a cutoff depth. Minimax alone did not satisfy the 5 second restriction in
            deciding on a move because it expanded every single branch, at every depth, until the terminal state.
            Alpha-beta pruning improved minimax by limiting how many branches needed to be fully explored,
            based on the utility value of at least one fully explored branch.
            However, alpha-beta pruning was still not adhering to the 5 second restriction on the full board,
            so an artificial cut-off limit was applied to resolve this problem.
        b) We ordered the successor states and also applied an artificial cutoff to the branching factor to
            limit how many positions we consider to be returned as the optimal move. We do this by sorting and
            extracting a fixed number of positions based on the layer number.
            - Our player agent is the MAX player and will try to maximise the utility of MIN layer;
                The positions with the smallest accumulated distance from each existing position is most likely to result
                in the highest utility value (shortest distance from existing positions means we are more likely t
                o build more 3 or 4 in a rows - this is what our evaluation function will use to compute a utility) -
                thus the MIN layer will have the smallest distances ordered in ascending order. By doing so,
                MAX layer will only have to fully explore the first branch and can then maximise pruning on
                all the other child branches (i.e. prune more nodes on the MIN layer).
            - The opponent will play as the MIN player; This agent is concerned with its successor layer - the MAX layer
                - and will try to minimise the utility.
                The maximum distance, between the position MIN player is considering and the existing
                MIN player positions, will most likely result in the smallest utility value.
                Thus, MAX layer must contain the positions that have the highest accumulated distance
                from existing positions of MIN player, ordered in descending order.
        c) Our heuristic is based on the number of 4, 3 and 2 in line made by a player in the current state of the board.
            Then in order to calculate the total score of the current state we apply weights 100, 5, and 1 to 4, 3 and 2
            in a line, respectively - applying a higher weight to a greater threat (4 pieces is closer to terminal state
            than 3 in a line making it a greater threat).

Algorithms experimented/Stages of implementation:
    - Minimax did not size up after a limit of a 3x3 board with goal of making 3 in a row.
    - With alpha-beta pruning, without a cutoff, the best we were able to implement was a game which made 3 in a row
        on an 8x8 board. The heuristic we used evaluated a state to return a value (10, 0, -10 respectively)
        based on whether the terminal state resolved to a win, draw, or lose.
    - To be able to size up the alpha-beta pruning algorithm to the full size of the board, we applied an artificial
        cut-off depth and a different utility function. The highest depth we were able to go to was depth 2.
        The heuristic/utility function was changed  to determine a score
        based on a weighted count of 4, 3, and 2 in a row.
    - To make our agent more intelligent, and increase its chances of winning, we applied a fixed opening move
        (in both cases of when we are either the first or second player)
        and blocking functions to have a more defensive approach.
    - Finally, to allow for better planning, we needed to find a way to be able to go deeper and expand
        the most promising nodes in the search tree. To do so, we needed to sort the nodes
        in order of expected utility based on distance of each legal move to existing positions or a player.
        The sorted list meant we would maximise pruning on all branches after fully exploring the first branch
        because we do not expect the successor branches to have a higher utility. To be able to further
        go down a deeper depth, we also apply an artificial cutoff to the branching factor and only consider the states
        that would result in the highest utilities.
            - through experimentation we have found that artificial cutoff depth 4 with 10 elements per depth gave
            the best result because the player is able to make a move in the 5 seconds window given and going to depth 4
            and exploring the 10 most promising nodes from the sorted list mentioned above always provides the best move
            for the given board in favor of the current player.
"""


import numpy as np

from gomokuAgent import GomokuAgent
from misc import winningTest, legalMove

DEPTH = 0


class Player(GomokuAgent):
    def move(self, board):
        l_moves = self.actions(board)

        if len(l_moves) == 121 or len(l_moves) == 120:
            if board[5, 5] == 0:
                return 5, 5
            else:
                return 4, 5
        else:
            win, moveLock = self.immediate_win_position(board, l_moves)
            if win:
                return moveLock

            block_closed_4, moveLock = self.block_closed_four(board, l_moves)
            if block_closed_4:
                return moveLock

            block_open_3, moveLock = self.block_open_three(board, l_moves)
            if block_open_3:
                return moveLock

            block_to_doubly_open_4, moveLock = self.block_to_doubly_open_four(board, l_moves)
            if block_to_doubly_open_4:
                return moveLock

            moveLock = self.alpha_beta_search(board, DEPTH, l_moves)
            return moveLock

    """ 
    Checks if it is possible to place a stone such that the player wins with his next move.
    And return the position to place the stone if possible, otherwise return False.
    """
    def immediate_win_position(self, board, l_moves):
        for pos in l_moves:
            copy = self.result(board, pos, self.ID)
            if winningTest(self.ID, copy, self.X_IN_A_LINE):
                return True, pos
        return False, None

    """
    Checks whether the opponent has formed four in a row with one side blocked by player's piece and the other
    side is open, then return the open position as the player's move with the aim of blocking the opponent from 
    making five in a row and reaching a terminal state 
    """
    def block_closed_four(self, board, l_moves):
        for pos in l_moves:
            copy = self.result(board, pos, -self.ID)
            if winningTest(-self.ID, copy, self.X_IN_A_LINE):
                return True, pos
        return False, None

    """
    For each legal move, check whether the move will extend the opponent's existing open three to an open four in a line,
    then return one of the open positions as the current player's next move in order to block the opponent.
    """
    def block_open_three(self, board, l_moves):
        for pos in l_moves:
            dictionary = {"fl1": True, "fl2": True, "fl3": True, "fl4": True}
            copy = self.result(board, pos, -self.ID)
            # y, x-1 row right
            for i in range(1, 4):
                if pos[1] + i > self.BOARD_SIZE-1 or copy[pos[0], pos[1] + i] != -self.ID:
                    dictionary["fl1"] = False
                    break
            # y+1, x column down
            for j in range(1, 4):
                if pos[0] + j > self.BOARD_SIZE-1 or copy[pos[0] + j, pos[1]] != -self.ID:
                    dictionary["fl2"] = False
                    break
            # y+1, x+1 diagonal down left to right
            for m in range(1, 4):
                if pos[0] + m > self.BOARD_SIZE-1 or pos[1] + m > self.BOARD_SIZE-1 or copy[pos[0] + m, pos[1] + m] != -self.ID:
                    dictionary["fl3"] = False
                    break
            # y+1, x-1 diagonal down right to left
            for t in range(1, 4):
                if pos[0] + t > self.BOARD_SIZE-1 or pos[1] - t < 0 or copy[pos[0] + t, pos[1] - t] != -self.ID:
                    dictionary["fl4"] = False
                    break

            if dictionary["fl1"]:
                if (pos[0], pos[1] + 4) in l_moves:
                    return True, pos
            if dictionary["fl2"]:
                if (pos[0] + 4, pos[1]) in l_moves:
                    return True, pos
            if dictionary["fl3"]:
                if (pos[0] + 4, pos[1] + 4) in l_moves:
                    return True, pos
            if dictionary["fl4"]:
                if (pos[0] + 4, pos[1] - 4) in l_moves:
                    return True, pos

        return False, None

    """
    For each legal move, check whether the move will extend the opponent's existing closed or broken three 
    to an open four in a line, then return one of the open positions as the current player's next move 
    in order to block the opponent. 
    """
    def block_to_doubly_open_four(self, board, l_moves):
        for pos in l_moves:
            dictionary = {"flag1": True, "flag2": True, "flag3": True, "flag4": True, "flag5": True, "flag6": True, "flag7": True, "flag8": True}
            copy = self.result(board, pos, -self.ID)
            # shift column of current position to the right by one - (y, x+1)
            for i in range(1, 4):
                if pos[1] + i > self.BOARD_SIZE-1 or copy[pos[0], pos[1] + i] != -self.ID:
                    if pos[1] + i < self.BOARD_SIZE and pos[1]+i+1 < self.BOARD_SIZE and copy[pos[0], pos[1]+i+1] == -self.ID and i == 3:
                        if (pos[0], pos[1] + i) in l_moves:
                            return True, (pos[0], pos[1] + i)
                        else:
                            dictionary["flag1"] = False
                            break
                    else:
                        dictionary["flag1"] = False
                        break
            # shift column of current position to the left by one - (y, x-1)
            for n in range(1, 4):
                if pos[1] - n < 0 or copy[pos[0], pos[1] - n] != -self.ID:
                    if pos[1] - n >= 0 and pos[1]-n-1 >= 0 and copy[pos[0], pos[1]-n-1] == -self.ID and n == 3:
                        if (pos[0], pos[1] - n) in l_moves:
                            return True, (pos[0], pos[1] - n)
                        else:
                            dictionary["flag2"] = False
                            break
                    else:
                        dictionary["flag2"] = False
                        break
            # shift current position to south-east position(diagonally down and to the right) - (y+1, x+1)
            for m in range(1, 4):
                if pos[0] + m > self.BOARD_SIZE-1 or pos[1] + m > self.BOARD_SIZE-1 or copy[pos[0] + m, pos[1] + m] != -self.ID:
                    if pos[0] + m < self.BOARD_SIZE and pos[1] + m < self.BOARD_SIZE and pos[0] + m + 1 < self.BOARD_SIZE and pos[1] + m + 1 < self.BOARD_SIZE and copy[pos[0] + m + 1, pos[1] + m + 1] == -self.ID and m == 3:
                        if (pos[0] + m, pos[1] + m) in l_moves:
                            return True, (pos[0] + m, pos[1] + m)
                        else:
                            dictionary["flag3"] = False
                            break
                    else:
                        dictionary["flag3"] = False
                        break
            # shift current position to south-west position(diagonally down and to the left) - (y-1, x-1)
            for r in range(1, 4):
                if pos[0] - r < 0 or pos[1] - r < 0 or copy[pos[0] - r, pos[1] - r] != -self.ID:
                    if pos[0] - r >= 0 and pos[1] - r >= 0 and pos[0] - r - 1 >= 0 and pos[1] - r - 1 >= 0 and copy[pos[0] - r - 1, pos[1] - r - 1] == -self.ID and r == 3:
                        if (pos[0] - r, pos[1] - r) in l_moves:
                            return True, (pos[0] - r, pos[1] - r)
                        else:
                            dictionary["flag4"] = False
                            break
                    else:
                        dictionary["flag4"] = False
                        break
            # shift row of current position down by one - (y+1, x)
            for j in range(1, 4):
                if pos[0] + j > self.BOARD_SIZE-1 or copy[pos[0] + j, pos[1]] != -self.ID:
                    if pos[0] + j < self.BOARD_SIZE and pos[0]+j+1 < self.BOARD_SIZE and copy[pos[0]+j+1, pos[1]] == -self.ID and j == 3:
                        if (pos[0] + j, pos[1]) in l_moves:
                            return True, (pos[0] + j, pos[1])
                        else:
                            dictionary["flag5"] = False
                            break
                    else:
                        dictionary["flag5"] = False
                        break
            # shift row of current position up by one - (y-1, x)
            for p in range(1, 4):
                if pos[0] - p < 0 or copy[pos[0] - p, pos[1]] != -self.ID:
                    if pos[0] - p >= 0 and pos[0]-p-1 >= 0 and copy[pos[0]-p-1, pos[1]] == -self.ID and p == 3:
                        if (pos[0] - p, pos[1]) in l_moves:
                            return True, (pos[0] - p, pos[1])
                        else:
                            dictionary["flag6"] = False
                            break
                    else:
                        dictionary["flag6"] = False
                        break
            # shift current position to north-east position(diagonally up and to the right) - (y-1, x+1)
            for k in range(1, 4):
                if pos[0] - k < 0 or pos[1] + k > self.BOARD_SIZE-1 or copy[pos[0] - k, pos[1] + k] != -self.ID:
                    if pos[0] - k >= 0 and pos[1] + k < self.BOARD_SIZE and pos[0] - k - 1 >= 0 and pos[1] + k + 1 < self.BOARD_SIZE and copy[pos[0] - k - 1, pos[1] + k + 1] == -self.ID and k == 3:
                        if (pos[0] - k, pos[1] + k) in l_moves:
                            return True, (pos[0] - k, pos[1] + k)
                        else:
                            dictionary["flag7"] = False
                            break
                    else:
                        dictionary["flag7"] = False
                        break
            # shift current position to north-west position(diagonally up and to the left) - (y+1, x-1)
            for t in range(1, 4):
                if pos[0] + t > self.BOARD_SIZE-1 or pos[1] - t < 0 or copy[pos[0] + t, pos[1] - t] != -self.ID:
                    if pos[0] + t < self.BOARD_SIZE and pos[1] - t >= 0 and pos[0] + t + 1 < self.BOARD_SIZE and pos[1] - t - 1 >= 0 and copy[pos[0] + t + 1, pos[1] - t - 1] == -self.ID and t == 3:
                        if (pos[0] + t, pos[1] - t) in l_moves:
                            return True, (pos[0] + t, pos[1] - t)
                        else:
                            dictionary["flag8"] = False
                            break
                    else:
                        dictionary["flag8"] = False
                        break

            if dictionary["flag1"] or dictionary["flag2"] or dictionary["flag3"] or dictionary["flag4"] or \
                    dictionary["flag5"] or dictionary["flag6"] or dictionary["flag7"] or dictionary["flag8"]:
                return True, pos
        return False, None

    """
    Alpha-beta search finds the most optimal next move for the player to make by planning and predicting a fixed 
    number - determined by the cutoff depth - of successor moves  
    """
    def alpha_beta_search(self, board, depth, l_moves):

        p = self.ID  # return the current player MAX(1) or MIN(-1)

        def max_value(board, alpha, beta, player, d, legal_moves):
            if self.cutoff_test(board, d):
                return self.utility(board)
            v = -np.inf
            copy = legal_moves

            # Sort list of distances in ascending order and expand the nodes with the N smallest values
            sorted_list = self.list_distances(board, legal_moves)
            sorted_list.sort()
            for m in sorted_list[:11]:
                copy.remove(m[1])
                v = max(v, min_value(self.result(board, m[1], -player), alpha, beta, -player, d + 1, copy))
                copy.append(m[1])
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(board, alpha, beta, player, d, legal_moves):
            if self.cutoff_test(board, d):
                return self.utility(board)
            v = np.inf
            copy = legal_moves

            # Sort list of distances in descending order and expand the nodes with the N largest values
            sorted_list = self.list_distances(board, legal_moves)
            sorted_list.sort(reverse=True)
            for m in sorted_list[:11]:
                copy.remove(m[1])
                v = min(v, max_value(self.result(board, m[1], -player), alpha, beta, -player, d + 1, copy))
                copy.append(m[1])
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Body of alpha_beta_search:
        alpha = -np.inf
        beta = np.inf
        best_move = None  # moveLock

        copy = l_moves

        # Sort list of distances in ascending order and expand the nodes with the N smallest values
        sorted_list = self.list_distances(board, l_moves)
        sorted_list.sort()
        for move in sorted_list[:11]:
            copy.remove(move[1])
            v = min_value(self.result(board, move[1], p), alpha, beta, p, depth + 1, copy)
            copy.append(move[1])
            if v > alpha:
                alpha = v
                best_move = move[1]
        return best_move

    """ 
    It gives all the legal moves (places not taken) from the given board as parameter
    """
    def actions(self, board):
        moves = []
        for x in range(0, self.BOARD_SIZE):
            for y in range(0, self.BOARD_SIZE):
                if legalMove(board, (x, y)):
                    moves.append((x, y))
        return moves

    """
    Outputs a list of accumulated distances from each legal position to each of current player's existing positions
    """
    def list_distances(self, board, l_moves):
        sorted_list = []
        for lmove in l_moves:
            sum = 0
            for mmove in self.my_pieces(board):
                distance = np.sqrt(((lmove[0]-mmove[0])**2)+((lmove[1]-mmove[1])**2))
                sum += distance
            sorted_list.append((sum, lmove))
        return sorted_list

    """
    Outputs a list of the current player's existing values
    """
    def my_pieces(self, board):
        my_pieces = []
        for x in range(0, self.BOARD_SIZE):
            for y in range(0, self.BOARD_SIZE):
                if board[x, y] == self.ID:
                    my_pieces.append((x, y))
        return my_pieces

    """
    Outputs the successor state for a given move on the current board
    """
    def result(self, board, move, player):
        copy = board.copy()
        copy[move] = player
        return copy

    """
    Returns evaluation score for each state based on the number of 2, 3 and 4 in a line made
    """
    def utility(self, board):
        count_two, count_three, count_four = self.evaluate(board)

        utility = 100*count_four + 5*count_three + 1*count_two

        return utility

    """
    Returns the count of 2, 3 and 4 in a line, for each direction, in a given state of the board
    """
    def evaluate(self, board):
        rotated_board = np.rot90(board)

        count4 = self.rowTest_count(board, rotated_board, 4) + self.diagTest_count(board, rotated_board, 4)
        count3 = self.rowTest_count(board, rotated_board, 3) + self.diagTest_count(board, rotated_board, 3)
        count2 = self.rowTest_count(board, rotated_board, 2) + self.diagTest_count(board, rotated_board, 2)

        return count2, count3, count4

    """
    Checks to see whether the current player or opponent has reached to terminal state and won the game or if the board
    has been filled
    """
    def terminal_test(self, board):
        flag = False

        if winningTest(self.ID, board, self.X_IN_A_LINE) or winningTest(-self.ID, board, self.X_IN_A_LINE):
            flag = True

        if len(self.actions(board)) == 0:
            flag = True

        return flag

    """
    Checks to see if we have reached a predetermined cutoff depth or if a terminal state has been found
    """
    def cutoff_test(self, board, depth):
        flag = False

        if depth == 4 or self.terminal_test(board):
            flag = True

        return flag

    """ Going through each row and column, and keep a count of all X_IN_A_LINE occurrences """
    def rowTest_count(self, board, rot_board, X_IN_A_LINE):
        count = 0
        count2 = 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE-X_IN_A_LINE+1):
                flag = True
                flag2 = True
                for i in range(X_IN_A_LINE):
                    if board[r, c+i] != self.ID:
                        flag = False
                        break
                if flag:
                    count += 1

                for j in range(X_IN_A_LINE):
                    if rot_board[r, c+j] != self.ID:
                        flag2 = False
                        break
                if flag2:
                    count2 += 1
        return count + count2

    """ Going through each diagonal of the board and keep a count of all X_IN_A_LINE occurrences """
    def diagTest_count(self, board, rot_board, X_IN_A_LINE):
        count = 0
        count2 = 0
        for r in range(self.BOARD_SIZE - X_IN_A_LINE + 1):
            for c in range(self.BOARD_SIZE - X_IN_A_LINE + 1):
                flag = True
                flag2 = True
                for i in range(X_IN_A_LINE):
                    if board[r + i, c + i] != self.ID:
                        flag = False
                        break
                if flag:
                    count += 1
                for j in range(X_IN_A_LINE):
                    if rot_board[r + j, c + j] != self.ID:
                        flag2 = False
                        break
                if flag2:
                    count2 += 1

        return count + count2
