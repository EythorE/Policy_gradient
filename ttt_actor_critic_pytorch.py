import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
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
    torch.manual_seed(seed)
    np.random.seed(seed)
reset_graph()


# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 9, 50, 1

actor = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax(dim=0),
)
critic = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Tanh(),
)

def get_action_and_value(actor, boards):
    boards = torch.from_numpy(boards).float()
    possible_actions_probs = actor(boards)
    action = int(torch.multinomial(possible_actions_probs.view(1,-1), 1))
    action_value = possible_actions_probs[action]
    return action, action_value
    
def get_action_value(actor, boards, action):
    boards = torch.from_numpy(boards).float()
    possible_actions_probs = actor(boards)
    action_value = possible_actions_probs[action]
    return action_value

def get_action(actor, boards):
    boards = torch.from_numpy(boards).float()
    possible_actions_probs = actor(boards)
    action = torch.multinomial(possible_actions_probs.view(1,-1), 1)
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
        
        
gamma = .90
actor_alpha = 0.05
critic_alpha = 0.05

critic_lambda = 0.9
actor_lambda = 0.9

critic_Z = [0 for layer in critic.parameters()]
actor_Z = [0 for layer in actor.parameters()]

forever = 100000

plt_iter = 1000
rew = []
rew_plt = []

from time import time
tic = time()

env = tictactoe()

for episode in range(forever):      
    state = env.reset()
    done = False
    I = 1
    step = 1
    
    while not done:
        #action = epsilon_greedy(critic, possible_boards) # Only use after_state values
        #pi = get_action_value(actor, possible_boards, action)
        
        possible_moves, possible_boards = env.legal_moves(1)
        action, pi = get_action_and_value(actor, possible_boards) # Using actor
        
        # calc actor gradient
        pi.clamp(min=1e-8) # so that log does not become nan
        log_pi = torch.log(pi) 
        actor.zero_grad()
        log_pi.backward()
        with torch.no_grad():
            for i, param in enumerate(actor.parameters()):
                actor_Z[i] = actor_lambda * I * actor_Z[i] + param.grad
        
        
        after_state, reward, done = env.step(possible_moves[action])
        
        if not done:
            value = get_state_value(critic, after_state)
            #calc critic gradient
            critic.zero_grad()
            value.backward()
            with torch.no_grad():
                for i, param in enumerate(critic.parameters()):
                    critic_Z[i] = critic_lambda * critic_Z[i] + param.grad
        else:
            value = 0
            
        with torch.no_grad():
            # other players move
            if not done:
                next_state, reward, done = env.make_move()
                reward = -reward
                next_value = get_state_value(critic, next_state)
            else:
                next_value = 0
                
            if step > 1:
                delta = reward + gamma*value - old_value
            
            old_value = value
        
            ###### plot
            if episode%plt_iter == 0:
                env.render()
                if done:
                    print('Reward: ',reward)
                    rew_plt.append(np.mean(np.equal(rew,-1)))
                    rew = []
                    plt.plot(rew_plt)
                    plt.show()
                    rnd = False
                    print("Episode: {}".format(episode))
                    toc=time()
                    print('time per',plt_iter,':',toc-tic)
                    tic=toc
            ######
            
            # apply gradients
            if step > 1:
                for i, param in enumerate(critic.parameters()):
                    param += critic_alpha * delta * critic_Z[i]
                for i, param in enumerate(actor.parameters()):
                    param += actor_alpha * delta * actor_Z[i]
 

#        if step > 1:
#            # apply gradients
#            value = get_state_value(critic, after_state)
#            critic.zero_grad()
#            value.backward()
#            with torch.no_grad():
#                for param in critic.parameters():
#                    param += critic_alpha * delta * param.grad
#            
#            pi = get_action_value(actor, possible_boards, action)
#            pi.clamp(min=1e-8) # so that log does not become nan
#            log_pi = torch.log(pi) 
#            actor.zero_grad()
#            log_pi.backward()
#            with torch.no_grad():
#                for param in actor.parameters():
#                    param += actor_alpha * I * delta * param.grad
            
        I *= gamma
        step +=1
        
    rew.append(reward)


