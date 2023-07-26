from puissance4 import *
def evaluate(board, player1):
    evaluationTable = [[3, 4, 5, 7, 5, 4, 3],
                      [ 4, 6, 8, 10, 8, 6, 4],
                      [ 5, 8, 11, 13, 11, 8, 5],
                      [ 5, 8, 11, 13, 11, 8, 5],
                      [ 4, 6, 8, 10, 8, 6, 4],
                      [ 3, 4, 5, 7, 5, 4, 3]]
    if player1=="O":
        player2 = "X"
    else:
        player2="O"
    utility = 138
    # check first if connections with size 4 exist
    if check(board, player1, 4) > 0:
        if check(board, player2, 4) > 0:
            return 0
        return 2*utility
    if check(board, player2, 4) > 0:
        return -2*utility
    # if not, use heuristic
    sum = 0
    for i in [0,1,2,3,4,5]:
        for j in [0,1,2,3,4,5,6]:
            if (board[i][j] == player1):
                sum += evaluationTable[i][j]
            else:
                if (board[i][j] == player2):
                    sum -= evaluationTable[i][j]
    return sum
    #return utility + sum

'''
Started as MiniMax but is an Alpha-Beta search variant now.
Old version but mostly defeats the new implemented Alpha-Beta Negamax Algorithm probably because of the evaluation.
'''
def minimax(board, depth, player, maximizingPlayer=True, alpha=-9999, beta=9999):
    # get all possible steps
    steps = poss_steps(board)
    # evaluate if endstate reached
    if depth==0 or steps==[]:
        wert = evaluate(board, player)
        return [wert,-1]
    if maximizingPlayer:
        maxValue = alpha
        bestStep = -1
        if player=="X":
            next_player = "O"
        else:
            next_player= "X"
        for step in steps:
            insert(board, step, player)
            #print "schritt: " + str(step) + " bei min"
            # QUICK EVALUATION if 4 CONNECTED then no need for further computation
            if check(board, next_player, 4) > 0:
                uninsert(board, step, player)
                return [999,step]
            val = minimax(board, depth-1, next_player, False, maxValue, beta)[0]
            uninsert(board, step, player)
            if val > maxValue:
                bestStep= step
                maxValue=val
                #alphabeta ergänzung
                if maxValue >= beta:
                    break
            #bestValue = max(bestValue, val)
            #print "bestVal: " + str(bestValue)
        return [maxValue, bestStep]
    else:
        #bestValue = 999999
        minValue = beta
        bestStep = -1
        if player=="X":
            next_player = "O"
        else:
            next_player= "X"
        for step in steps:
            insert(board, step, player)
            #print "schritt: " + str(step) + " bei max"
            # QUICK EVALUATION if 4 enemy CONNECTED NO NEED for further computation
            if check(board, player, 4) > 0:
                uninsert(board, step, player)
                return [-999,step]
            val = minimax(board, depth-1, next_player, True, alpha, minValue)[0]
            uninsert(board, step, player)
            if val < minValue:
                bestStep= step
                minValue=val
                if minValue <= alpha:
                    break
            #bestValue = min(bestValue, val)
            #print "bestVal: " + str(bestValue)
        return [minValue, bestStep]

def uninsert(board, col, symbol):
    for row in [1,2,3,4,5,6]:
        if (board[row-1][col-1] == symbol):
            board[row-1][col-1] = " "
            break

def poss_steps(board):
    ret = []
    for i in range(len(board[1])):
        if board[0][i]==" ":
            ret.append(i)
    return map(lambda x:x+1, ret)

'''
Checks for player sp how many connections of size anz he has.
'''
def check(feld, sp, anz):
    breite = len(feld[0])
    hoehe = len(feld)
    ret = 0

    #waagerecht
    for i in range(hoehe):
        tmp = 0
        for j in range(breite):
            if feld[i][j]==sp:
                tmp = tmp + 1
            else:
                tmp = 0
            if tmp>=anz:
                    ret = ret + 1
    #senkrecht
    for i in range(breite):
        tmp = 0
        for j in range(hoehe):
            if feld[j][i]==sp:
                tmp = tmp + 1
            else:
                tmp = 0
            if tmp>=anz:
                    ret = ret + 1

    #schräg rechts hoch

    sub=anz-1
    for i in range(hoehe-1,sub-1,-1):
        for j in range(0,breite-sub,1):
            tmp = 0
            for t in range(anz):
                if(feld[i-t][j+t]==sp): tmp = tmp+1
            if tmp == anz: ret = ret + 1

    #schräg links
    for i in range(hoehe-1,sub-1,-1):
        for j in range(breite-1,sub-1,-1):
            tmp = 0
            for t in range(anz):
                if(feld[i-t][j-t]==sp): tmp = tmp+1
            if tmp == anz: ret = ret + 1

    return ret

def insert(board, col, symbol):
    valid_move = False
    if(1 <= col <= 7):
        while not valid_move:
            for row in range (6,0,-1):
                if (1 <= row <= 6) and (board[row-1][col-1] == " "):
                    board[row-1][col-1] = symbol
                    return True
    else:
        print ("Sorry, invalid input. Please try again!\n")
    return False

'''
Insert and return a copy of the Game.
'''
def insert2(board, col, symbol):
    board_copy = dupl(board)
    if(insert(board_copy,col,symbol)):
        return board_copy
    return []
