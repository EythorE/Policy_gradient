import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython.display import clear_output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class tictactoe():
    def __init__(self):
        self.board = np.zeros(9, dtype='int')
        
    def reset(self):
        self.board[:] = 0
        self.done = False
        return np.copy(self.board)
    
    def legal_moves(self, player):
        moves = np.where(self.board == 0)[0]
        boards = []
        for move in moves:
            board = np.copy(self.board)
            board[move] = player
            boards.append(board)
        return moves, np.array(boards, dtype='double')
    
    def swap_player(self):
        self.board = -self.board
    
    # oppents random move
    def make_move(self, player=-1):
        moves, _ = self.legal_moves(player)
        return self.step(np.random.choice(moves, 1),player)
    
    def step(self, move, player=1):
        assert self.board[move] == 0, "Tried to play an illegal move, player = %d"%player
        assert not self.done, "Game has finished must call tictactoe.reset()"
        self.board[move] = player
        reward = 0
        self.done = False
        if self.iswin(player):
            reward = 1
            self.done = True
        if not np.any(self.board==0):
            self.done = True
        return np.copy(self.board), reward, self.done
        
    def iswin(self, player):
        for i in range(3):
            if np.all(self.board[[i*3, i*3+1, i*3+2]]==player) | np.all(self.board[[i, i+3, i+6]]==player):
                return True
        if np.all(self.board[[0, 4, 8]] == player) | np.all(self.board[[2, 4, 6]] == player):
            return True
        return False
        
    def render(self):
        data_mat = self.board.reshape(3, 3)
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                token = ""
                if data_mat[i, j] == 1:
                    token = 'x'
                if data_mat[i, j] == 0:
                    token = ' '
                if data_mat[i, j] == -1:
                    token = 'o'
                out += token + ' | '
            print(out)
        print('-------------')
        

def reset_graph(seed=42):
    #tf.reset_default_graph()
    #tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()


# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 9, 50, 1

actor = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
critic = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Tanh(),
)
softmax = torch.nn.Softmax(dim=0)
sigmoid = torch.nn.Sigmoid()

def get_action_value(actor, board):
    board = torch.from_numpy(board).float()
    action_value = actor(board)
    action_value = sigmoid(action_value)
    return action_value

def get_action(actor, boards):
    boards = torch.from_numpy(boards).float()
    possible_actions_values = actor(boards)
    possible_actions_props = softmax(possible_actions_values)
    action = torch.multinomial(possible_actions_props.view(1,-1), 1)
    return int(action)

def get_state_value(critic, after_state):
    after_state = torch.from_numpy(after_state).float()
    value = critic(after_state)
    return value

def epsilon_greedy(critic, possible_boards, epsilon=.9):
    possible_boards = torch.from_numpy(possible_boards).float()
    values = critic(possible_boards)
    if np.random.random()<epsilon:
        _ , index = values.max(0)
    else:
        index = np.random.randint(0, len(possible_boards))
    return int(index)
        
        
    

env = tictactoe()


gamma = .90
actor_alpha = 0.05
critic_alpha = 0.1
forever = 100000

plt_iter = 1000
rew = []
rew_plt = []


for episode in range(forever):
    
    ###### output
    if episode%1000==0:
        print("\rEpisode: {}".format(episode, end=""))
        if episode%plt_iter == 0:
            clear_output()
            print("\rIteration: {}".format(episode))
    #####        
    
            
    state = env.reset()
    done = False
    possible_moves, possible_boards = env.legal_moves(1)
    I = 1
    while not done:
        with torch.no_grad():
            action = get_action(actor, possible_boards) # Using actor
            #action = epsilon_greedy(critic, possible_boards) # Only use after_state values
            after_state, reward, done = env.step(possible_moves[action])
            if not done:
                value = get_state_value(critic, after_state)
            else:
                value = 0
            # other players move
            if not done:
                next_state, reward, done = env.make_move()
                reward = -reward
                possible_moves, possible_boards = env.legal_moves(1)
                next_value = get_state_value(critic, next_state)
            else:
                next_value = 0
            delta = reward + gamma*next_value - value
        
        ###### plot
        if episode%plt_iter == 0:
            env.render()
            if done:
                print('Reward: ',reward)
                rew_plt.append(np.mean(np.equal(rew,1)))
                rew = []
                plt.plot(rew_plt)
                plt.show()
                rnd = False
        ######
                
        # apply gradients
        value = get_state_value(critic, after_state)
        critic.zero_grad()
        value.backward()
        with torch.no_grad():
            for param in critic.parameters():
                param += critic_alpha * delta * param.grad
        
        pi = get_action_value(actor, after_state)
        log_pi = torch.log(pi) 
        actor.zero_grad()
        log_pi.backward()
        with torch.no_grad():
            for param in actor.parameters():
                param += actor_alpha * delta * param.grad
            
        I *= gamma
        
    rew.append(reward)


