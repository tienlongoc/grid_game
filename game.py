#!/usr/bin/python3

from termcolor import colored

class Player:
    def __init__(self, player_number, hand, color):
        self.hand = hand
        self.color = color
        self.player_number = player_number
        self.score = 0
    def __repr__(self):
        return str(self.player_number)


class Node:
    def __init__(self):
        self.value = 0
        self.owner = Player(0, [], 'white')
        self.adjacent_nodes = []
    def add_adjacent_node(self, node):
        if node not in self.adjacent_nodes:
            self.adjacent_nodes.append(node)
        if self not in node.adjacent_nodes:
            node.adjacent_nodes.append(self)
    def __repr__(self):
        return colored(self.value, self.owner.color)

class Board:
    def __init__(self, dimension):
        self.dimension = dimension
        self.board = [[Node() for _ in range(dimension)] for _ in range(dimension)]
        for i in range(dimension):
            for j in range(dimension):
                if i != dimension-1:
                    self.board[i][j].add_adjacent_node(self.board[i+1][j])
                if j != dimension-1:
                    self.board[i][j].add_adjacent_node(self.board[i][j+1])

    def print_board(self):
        print(self.board[0][0], '|', self.board[0][1], '|', self.board[0][2])
        print('- - - - -')
        print(self.board[1][0], '|', self.board[1][1], '|', self.board[1][2])
        print('- - - - -')
        print(self.board[2][0], '|', self.board[2][1], '|', self.board[2][2])

    def validate_move(self, player, move):
        if len(move) == 0:
            return False
        elif len(move) != self.dimension:
            print('\nPlease enter', self.dimension, 'space-separated integers\n')
            return False
        for i in range(len(move)):
            try:
                move[i] = int(move[i])
            except ValueError:
                print('\nNon integer detected in move\n')
                return False
        if move[0] not in range(self.dimension) or move[1] not in range(self.dimension):
            print('\nOut of range coordinate specified\n')
            return False
        if self.board[move[0]][move[1]].owner.player_number != 0:
            print('\nCoordinate is already taken\n')
            return False
        if move[2] not in player.hand:
            print('\nCard selected is not in player hand\n')
            return False
        return True

    def make_move(self, player, move):
        node = self.board[move[0]][move[1]]
        node.owner = player
        node.value = move[2]
        player.score += 1
        for adjacent_node in node.adjacent_nodes:
            if adjacent_node.value != 0 and adjacent_node.value < node.value and adjacent_node.owner != player:
                adjacent_node.owner.score -= 1
                adjacent_node.owner = player
                player.score += 1
        player.hand.remove(move[2])


def game():
    player1 = Player(1, list(range(1,6)), 'red')
    player2 = Player(2, list(range(2,6)), 'blue')
    board = Board(3)

    turn = 1

    print('Begin game\n')

    current_player = player1

    while turn < 10:
        print('Turn: ' + str(turn))
        board.print_board()
        print('\n')
        print('Player 1\'s hand: ', player1.hand)
        print('Player 2\'s hand: ', player2.hand)
        print('\n')
        print('Player', current_player, '\'s turn')
        move = []
        while not board.validate_move(current_player, move):
            move = []
            print('Please select row coordinate')
            move.append(input())
            print('Please select column coordinate')
            move.append(input())
            print('Please select card to play')
            move.append(input())

        board.make_move(current_player, move)

        if current_player == player1:
            current_player = player2
        elif current_player == player2:
            current_player = player1
        turn += 1
        
    board.print_board()

    print('Player 1\'s score:', player1.score)
    print('Player 2\'s score:', player2.score)


def train(num_training_samples = 10):
    training_game_count = 0
    # x1 number of red on board
    # x2 number of blue on board
    # x3 sum of card in hand
    # x4 sum of card in opponent hand
    # x5 number of open file for your cards on board
    # x6 number of open files for opponent cards on board
    
    players_target_functino_weight_vectors = [[.5]*6,[.5]*6]
    game_status_count = [0,0]

    while training_game_count < num_training_samples:
        experiment_generator = ExperimentGenerator()




game()
