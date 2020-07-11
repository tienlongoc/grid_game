#!/usr/bin/python3

# Idea is implement my own grid game, and train the code to evaluate desired moves by using a linear function, with each variable representing some evaluated state of board, e.g. running score of players or how many open files a player has

# Reference used: https://gist.github.com/NoblesseCoder/cccf260ecc3e2052a0a1a013a7f7ac54



# Allow for arbitrary dimension of board
# player to select order of play and number of training iterations
# train by random moves
# more variables in weighting vecotr. Variable number of variables? so we don't have to hard code number of variables in strategy vector every time
# reduce to own code, without following framework so rigidl
# give training reports
# Try to reduce the number of times I have to define the game. Right now there wll be 3 times based on playing seleciton + 1 more time in trainer
# Things like calculate non-final/final score and extract features don't have to be exclusive in player class, we could probably make a parent class to do this. Or maybe put it all in the Board class?

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
        for row in range(self.dimension):
            row_to_print = ''
            for col in range(self.dimension):
                row_to_print += str(self.board[row][col])
                if col != self.dimension - 1:
                    row_to_print += ' | '
            print(row_to_print)
            if row != self.dimension - 1:
                print(('- ' * (self.dimension * 2 - 1))[:-1])

    def validate_move(self, player, move):
        if len(move) == 0:
            return False
        elif len(move) != 3:
            print('\nPlease enter 3 space-separated integers\n')
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

def human_game(dimension = 3):
    player1 = Player(1, list(range(1, (dimension**2 + 1)//2 + 1)), 'red')
    player2 = Player(2, list(range(2 - ((dimension + 1) % 2), (dimension**2 + 1)//2 + 1)), 'blue')
    board = Board(dimension)
    turn = 1

    print('Begin game\n')

    current_player = player1
    next_player = player2

    while turn < dimension**2+1:
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




class Trainer:
    def __init__(self, player_num_to_train, num_train = 100, dimension = 3):
        self.num_train = num_train
        self.dimension = dimension
        self.player_num_to_train = player_num_to_train
    
    # Corresponds to the "Performance System" in Tom Mitchell's model flow
    # We don't require an explicit "Experiment Generator" in our case, since we are happy to always state from the same board state (the empty board)
    # This "Performance System" will play the game starting from the initial board state, and return a history of each move, plus info on who won the game
    def play_training_game(self, player1, player2):
        game_history = []
        
        turn = 1
        num_turns = dimension ** 2

        board = Board(dimension)
        current_player = player1
        next_player = player2

        while turn <= num_turns:
            temp_board = copy.deepcopy(board)
            game_history.append(temp_board)
            chosen_move = current_player.choose_move(board, next_player)
            num_flipped_cards = board.make_move(current_player, chosen_move)
            current_player.carde += 1 + num_flipped_cards
            next_player.score -= num_flipped_cards
            current_player.hand.remove(chosen_move[2])
            current_player, next_player = next_player, current_player
            turn += 1

        final_game_status = [0,0,1]
        if self.player1.score > self.player2.score:
            final_game_status = [1,0,0]
        elif self.player1.score < self.player2.score:
            final_game_status = [0,1,0]

        return game_history, final_game_status

    # Corresponds to the "Critic" in Tom Mitchell's model flow
    # It looks at the game history, and it will try to tell the player how good the play was, and how we should adjust
    def critique_training_history(self, game_history, player_to_train, training_opponent, final_game_status):
        training_samples = []
        for turn in range(len(game_history)):
            if turn % 2 == player_to_train.player_number - 1: # Don't bother evaluating for the training opponent
                continue

            # We use the intermediate board state & our current approximation its the target value as the training value for the current target value function
            # This is equation 1.1 on Tom Mitchell's book
            feature_vector = player_to_train.extractFeatrues(game_history[turn], trianing_opponent)
            associated_score = player_to_train.compute_non_final_score(feature_vector)
            training_samples.append([feature_vector, associated_score])

        # Game has finished -- we can work out the absolute scores here
        feature_vector = player_to_train.extractFeatures(game_history[-1], training_opponent)
        associated_score = play_to_train.compute_final_score(final_game_status)
        training_samples.append([feature_vector, associated_score])

        return training_samples

    # Now that we have the training recommendation from the critic, we will use it to improve the target playing strategy function, by updating its linear weights
    # Improvement strategy is chosen to be LMS weight update
    # If we want, we can tweak alpha, which essentially means "how much adjustment I should make based on results of each game"
    def update_playing_strategy_vector(self, training_samples, player_to_train, alpha = 0.1):
        vector_to_update = player_to_train.playing_strategy_vector
        for i in range(len(training_samples)):
            training_score = training_samples[i+1][1]
            idealised_score = training_samples[i][1]
            training_feature_vector = training_samples[i][0]

            # Implementation of LMS weight update rule described in Tom Mitchell's text
            # w_ is the weight vector: this is the vector we want to use to guide playing strategy = vector_to_update
            # x_ is the feature vector: e.g. x_1 is player_to_train's score, etc = player_to_train.extract_features(board state b) = training_featrue_vector
            # We improve w_ thus -- for each step in game history (from player_to_train's point of view, i.e. opponent's turns are forgotten) with board state b:
            # w_ = w_ + alpha( V_train(b) - V^(b) )x_
            # Where V^(b) is the score of board state b computed using our current score evaluation (i.e. computed score using our playing_strategy_vector trained up until last game) = idealised_score
            # And V_train(b) is taken to be V^(b_successor) (we will use board state after opponent's made the move as b_successor, i.e. turn+2) = training_score
            # For the last turn before game end, we will take realised V^(b_successor) to be V_train(b) -- in this case, we know with certainty how the game ends, so we will use the game's end state to workout ternary options for V^(b_successor) (win/lose/draw), instead of using our current idealised calculation per score evaluation via playing_strategy_vector. This is the key weight adjustment that will eventually trickle down to better weights for playing_strategy_vector in relation to earlier game states.
            # I can see that this algorithm does help adjust our playing_strategy_vector based on a reward mechanism, but I'm not completely clear on why this would end up minimising the LMS error. Tom Mitchell says he'll discuss this aspect in chapter 4.
            vector_to_update = list(map(add, vector_to_update,  [x * alpha * (training_score - idealised_score) for x in training_feature_vector]))

        return vector_to_update


    def train(self):
        training_game_count = 0

        game_status_count = [0,0,0] # Player 1 wins, player 2 wins, draws

        playing_strategy_vector = ([1]*7)

        while training_game_count < self.num_train:
            if self.player_num_to_train == 1:
                player1 = Player(1, list(range(1, (self.dimension**2 + 1)//2 + 1)), 'red', playing_strategy_vector = playing_strategy_vector)
                player2 = Player(2, list(range(2 - ((self.dimension + 1) % 2), (self.dimension**2 + 1)//2 + 1)), 'blue', play_random_moves = True)
                player_to_train = player1
                training_opponent = player2
            else: # self.player_num_to_train == 2
                player1 = Player(1, list(range(1, (self.dimension**2 + 1)//2 + 1)), 'red', play_random_moves = True)
                player2 = Player(2, list(range(2 - ((self.dimension + 1) % 2), (self.dimension**2 + 1)//2 + 1)), 'blue', playing_strategy_vector)
                player_to_train = player2
                training_opponent = player1

            game_history, final_game_status = self.play_training_game(player1, player2)

            game_status_count = list(map(add, game_status_count, final_game_status))

            training_samples = critique_training_history(game_history, player_to_train, training_opponent, final_game_status)

            playing_strategy_vector = update_playing_strategy_vector(training_samples)

            training_game_count += 1

        print("Training games played:", num_train)
        print("Player 1 wins:", game_status_count[0])
        print("Player 2 wins:", game_status_count[1])
        print("Draws:", game_status_count[2])
        print("Final learnt playing strategy vector", playing_strategy_vector)

        return playing_strategy_vector






if __name__ == "__main__":
    play_selection = ''
    board_selection = ''
    num_train = ''
    while play_selection not in ['1','2','3']:
        play_selection = input("Please select an option:\n1: Human vs Human\n2: Human vs AI\n3: AI vs Human\n")
    while board_selection not in ['3','4','5','6','7']:
        board_selection = input("Please select board size (3-7)\n")
    board_selection = int(board_selection)
    while num_train == '' and play_selection != '1':
        num_train = input("Select number of training iterations for bot\n")
        try:
            num_train = int(num_train)
        except ValueError:
            num_train = ''
        if num_train < 1:
            num_train = ''
    if play_selection == '1':
        human_game(dimension = board_selection)
    elif play_selection == '2':
        player_strategy_vector = Trainer(2, num_train, board_selection).train()
        game_vs_player2(playing_strategy_vector = train(2, num_train, board_selection), dimension = board_selection)
    elif play_selection == '3':
        game_vs_player1(playing_strategy_vector = train(1, num_train, board_seletion), dimension = board_selection)

