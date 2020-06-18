#!/usr/bin/python3

# Idea is implement my own grid game, and train the code to evaluate desired moves by using a linear function, with each variable representing some evaluated state of board, e.g. running score of players or how many open files a player has

# Reference used: https://gist.github.com/NoblesseCoder/cccf260ecc3e2052a0a1a013a7f7ac54

from termcolor import colored
from numpy import random
import copy

def normalise(l):
    return [elem/sum(l) for elem in l]


class Player:
    def __init__(self, player_number, hand, color, target_function_weight_vector = normalise([1]*7)):
        self.hand = hand
        self.color = color
        self.player_number = player_number
        self.score = 0
        self.target_function_weight_vector = target_function_weight_vector

    def __repr__(self):
        return str(self.player_number)

    def find_legal_moves(self, board):
        legal_moves = []
        for row in range(len(board.board)):
            for col in range(len(board.board[row])):
                if board.board[row][col].value == 0:
                    for card in self.hand:
                        legal_moves.append([row,col,card])
        return legal_moves

    def extract_features_from_board(self, board, opponent):
        # I have chosen to use following parameters to model linear playing behaviour
        # x_1 self score
        # x_2 opponent score
        # x_3 sum of cards in hand
        # x_4 sum of cards in opponent hand
        # x_5 number of open file for your cards on board
        # x_6 number of open files for opponent cards on board
        x = [1]
        x.append(self.score)
        x.append(opponent.score)
        x.append(sum(self.hand))
        x.append(sum(opponent.hand))
        def num_open_files(player):
            open_files = 0
            for row in board.board:
                for node in row:
                    if node.owner == player:
                        open_files += sum([1 for adjacent_node in node.adjacent_nodes if adjacent_node.value == 0])
            return open_files
        x.append(num_open_files(self))
        x.append(num_open_files(opponent))
        return x

    def compute_non_final_score(self, feature_vector):
        # Compute move score based on our current approximation of target function weight vector
        # Maybe I want to normalise this? As the final score is normalised by 100.
        return sum([i*j for (i,j) in zip(self.target_function_weight_vector, feature_vector)])

    def compute_final_score(self, opponent):
        # When the game is finished, we can score player performance with absolute certainty. This will be used in back tracking lms weight updates.
        # draw = 0, win = 1, lose = -1
        # In current implementation, we are not calling this function.
        game_score = 0
        if self.score < opponent.score:
            game_score -= 100 
        elif self.score > opponent.score:
            game_score += 100
        return game_score

    def choose_move(self, board, opponent):
        legal_moves = self.find_legal_moves(board)
        legal_move_scores = []
        for move in legal_moves:
            tmp_board = copy.deepcopy(board)
            tmp_board.make_move(self, move)
            feature_vector = self.extract_features_from_board(tmp_board, opponent)
            legal_move_scores.append(self.compute_non_final_score(feature_vector))
#       return legal_moves[legal_move_scores.index(max(legal_move_scores))] # naive version of function that just chooses move with max score. Instead we will try to make a choice based on probabilities, rather than max score
        return legal_moves[random.choice(len(legal_moves), p = normalise(legal_move_scores))]



    def choose_random_move(self, board):
        # We could use this to play against a player who chooses random moves
        legal_moves = self.find_legal_moves(board)
        return random.choice(legal_moves)


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
        num_flipped = 0
        for adjacent_node in node.adjacent_nodes:
            if adjacent_node.value != 0 and adjacent_node.value < node.value and adjacent_node.owner != player:
                adjacent_node.owner = player
                num_flipped += 1
        return num_flipped



class ExperimentGenerator:
    # Generate new problems. We won't bother experimenting with mid-game training, will just start with empty board
    def __init__(self):
        self.init_board_state = Board(3)

    def generate_new_problem(self):
        return self.init_board_state

class PerformanceSystem:
    # Takes the given board state (in our case just start with empty board), play the game using the given target function weight vectors, and return the game history, to be further evaluated in next step
    def __init__(self, init_board_state, players_target_function_weight_vectors):
        self.board = init_board_state
        self.players_target_function_weight_vectors = players_target_function_weight_vectors
        self.player1 = Player(1, list(range(1,6)), 'red', players_target_function_weight_vectors[0])
        self.player2 = Player(2, list(range(2,6)), 'blue', players_target_function_weight_vectors[1])
        
    def generate_game_history(self):
        # Pit 2 training agents against each other
        game_history = []
        # Wanted to implement a way to read stdin prompts from game(), and reply to it -- but before I figure that out, let's just copy the game code here.
        turn = 1
        current_player = self.player1
        next_player = self.player2
        while turn < 10:
            temp_board = copy.deepcopy(self.board)
            game_history.append(temp_board)
            chosen_move = current_player.choose_move(self.board, next_player)
            num_flipped = self.board.make_move(current_player, chosen_move)
            current_player.score += 1 + num_flipped
            next_player.score -= num_flipped
            current_player.hand.remove(chosen_move[2])
            current_player, next_player = next_player, current_player
            turn += 1

        return game_history, [self.player1.score, self.player2.score]

class Critic:
    # Works through the game history, and returns historical list of feature vectors and associated scores. However at this point I can't see why we can't just do this as part of the performance system.
    def __init__(self, game_history, players_target_function_weight_vectors, final_scores):
        self.game_history = game_history
        self.players_target_function_weight_vectors = players_target_function_weight_vectors
        self.player1 = Player(1, list(range(1,6)), 'red', players_target_function_weight_vectors[0])
        self.player2 = Player(2, list(range(2,6)), 'blue', players_target_function_weight_vectors[1])
        self.final_scores = final_scores

    def compute_final_score(self, score1, score2):
        game_score = 0
        if score1 < score2:
            game_score -= 100 
        elif score1 > score2:
            game_score += 100
        return game_score

    def generate_training_samples(self):
        training_samples = [[],[]]
        current_player = self.player1
        next_player = self.player2
        for turn in range(len(self.game_history) - 1):
            # For each turn, add the current training feature vector, using information from our (approximated) target function. Notice the +1 index -- this obvioiusly makes a difference, but I don't really appreciate why we want to make this shift at this point.
            feature_vector = current_player.extract_features_from_board(self.game_history[turn+1], next_player)
            associated_score = current_player.compute_non_final_score(feature_vector)
            training_samples[turn % 2].append([feature_vector,associated_score])
            current_player, next_player = next_player, current_player
        # Game has finished -- we can work out absolute score here.
        player1_last_feature_vector = self.player1.extract_features_from_board(self.game_history[-1], self.player2)
        player1_last_associated_score = self.compute_final_score(*(self.final_scores))
        player2_last_feature_vector = self.player2.extract_features_from_board(self.game_history[-1], self.player1)
        player2_last_associated_score = self.compute_final_score(*(self.final_scores[::-1]))

        training_samples[0].append([player1_last_feature_vector, player1_last_associated_score])
        training_samples[1].append([player2_last_feature_vector, player2_last_associated_score])
        return training_samples

class Generaliser:
    # Takes training samples from Critic, and improves the target function. Improvement strategy is chosen to be LMS weight update here.
    def __init__(self, training_samples, players_target_function_weight_vectors):
        self.training_samples = training_samples
        self.players_target_function_weight_vectors = players_target_function_weight_vectors

    def compute_non_final_score(self, weight_vector, feature_vector):
        return sum([i*j for (i,j) in zip(weight_vector, feature_vector)])

    def lms_weight_update(self, alpha = 0.1):
        player1_target_function_weight_vector = self.players_target_function_weight_vectors[0]
        player2_target_function_weight_vector = self.players_target_function_weight_vectors[1]
        for training_sample in self.training_samples[0]: # Player 1
            training_score = training_sample[1]
            training_feature_vector = training_sample[0]
            predicted_score = self.compute_non_final_score(self.players_target_function_weight_vectors[0], training_feature_vector)          # Essentially the difference between the two is the last step. On the last training point, we have realised score value, and we will use that to work our regression on the whole weight vector.
            adjustment = alpha * (training_score - predicted_score)
            adjustment_vector = [v*adjustment for v in training_feature_vector]
            player1_target_function_weight_vector = normalise([sum(x) for x in zip(player1_target_function_weight_vector, adjustment_vector)])
        for training_sample in self.training_samples[1]:
            training_score = training_sample[1]
            training_feature_vector = training_sample[0]
            predicted_score = self.compute_non_final_score(self.players_target_function_weight_vectors[1], training_feature_vector)
            adjustment = alpha * (training_score - predicted_score)
            player2_target_function_weight_vector = normalise([sum(x) for x in zip(player2_target_function_weight_vector, adjustment_vector)])
        return [player1_target_function_weight_vector,player2_target_function_weight_vector]



def train(num_training_samples = 10):
    training_game_count = 0
    
    players_target_function_weight_vectors = [normalise([1]*7),normalise([50]*7)]
    game_status_count = [0,0]

    while training_game_count < num_training_samples:
        init_board_state = ExperimentGenerator().generate_new_problem()

        game_history, final_scores = PerformanceSystem(init_board_state, players_target_function_weight_vectors).generate_game_history()

        training_samples = Critic(game_history, players_target_function_weight_vectors, final_scores).generate_training_samples()

        players_target_function_weight_vectors = Generaliser(training_samples, players_target_function_weight_vectors).lms_weight_update()

        training_game_count += 1

    # Print out our result "AI" -- in form of learned weights from previous games
    print("Learnt weight vectors:", players_target_function_weight_vectors)

    return players_target_function_weight_vectors

def game(player1 = Player(1, list(range(1,6)), 'red'), player2 = Player(2, list(range(2,6)), 'blue'), board = Board(3)):
    turn = 1

    print('Begin game\n')

    current_player = player1
    next_player = player2

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
            move.append(input('Please select row coordinate\n'))
            move.append(input('Please select column coordinate\n'))
            move.append(input('Please select card to play\n'))

        num_flipped = board.make_move(current_player, move)
        current_player.score += 1 + num_flipped
        next_player.score -= num_flipped
        current_player.hand.remove(move[2])

        current_player, next_player = next_player, current_player
        turn += 1
        
    board.print_board()

    print('Player 1\'s score:', player1.score)
    print('Player 2\'s score:', player2.score)

def game_vs_player1(player1 = Player(1, list(range(1,6)), 'red'), player2 = Player(2, list(range(2,6)), 'blue'), board = Board(3), target_function_weight_vector = normalise([1]*7)):
    turn = 1

    print('Begin game\n')

    current_player = player1
    next_player = player2

    while turn < 10:
        print('Turn: ' + str(turn))
        board.print_board()
        print('\n')
        print('Player 1\'s hand: ', player1.hand)
        print('Player 2\'s hand: ', player2.hand)
        print('\n')
        print('Player', current_player, '\'s turn')
        move = []
        if current_player == player1:
            move = player1.choose_move(board, player2)
        elif current_player == player2:
            while not board.validate_move(current_player, move):
                move = []
                move.append(input('Please select row coordinate\n'))
                move.append(input('Please select column coordinate\n'))
                move.append(input('Please select card to play\n'))

        num_flipped = board.make_move(current_player, move)
        current_player.score += 1 + num_flipped
        next_player.score -= num_flipped
        current_player.hand.remove(move[2])

        current_player, next_player = next_player, current_player
        turn += 1
        
    board.print_board()

    print('AI\'s score:', player1.score)
    print('Your score:', player2.score)

if __name__ == "__main__":
    players_target_function_weight_vectors = train(500)
    selection = ''
    while selection not in ['1','2','3']:
        selection = input("Please select an option:\n1: Human vs Human\n2: Human vs AI\n3: AI vs Human\n")
    if selection == '1':
        game()
    elif selection == '2':
        game_vs_player2(target_function_weight_vector = players_target_function_weight_vectors[1])
    elif selection == '3':
        game_vs_player1(target_function_weight_vector = players_target_function_weight_vectors[0])




# improvements: there are a few functions that are referenced by multipled classes. Could make a parent class to make neater inheritance.
