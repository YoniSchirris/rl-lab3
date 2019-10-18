import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import seaborn as sns; sns.set()
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import random
import time
import pickle
from collections import defaultdict

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

EPS = float(np.finfo(np.float32).eps)

assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

import gym
env = gym.envs.make("CartPole-v0")

class QNetwork(nn.Module):
    """
    The Neural Network for the Q-Function.
    
    IN during initialization: 
    Number of states, number of actions, (number of hidden layers, optional)

    IN for a run:
    Vector describing the state, should be of length(number of states) given during initialization

    OUT:
    Value for each possible action
    """
    
    def __init__(self, num_s, num_a, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_s, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_a)

    def forward(self, x):
        # YOUR CODE HERE
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


# Let's instantiate and test if it works
num_hidden = 128
torch.manual_seed(1234)
model = QNetwork(num_hidden=num_hidden, num_a=2, num_s=4)

test_model = nn.Sequential(
    nn.Linear(4, num_hidden), 
    nn.ReLU(), 
    nn.Linear(num_hidden, 2)
)

x = torch.rand(10, 4)

class ReplayMemory:
    """
    Defines whether or not experience replay is used, tracks memory, and implements sampling function

    IN during initialization: max capacity of memory, (useTrick stating whether or not to use experience replay, optional)
    """
    
    def __init__(self, capacity, useTrick=True):
        self.capacity = capacity
        self.memory = []
        # useTrick defines whether or not experience replay is really used
        self.useTrick = useTrick

    def push(self, transition):
        # tracks memory
        if len(self.memory) >= self.capacity:
            # if memory is full we remove the last value and add the new value at beginning
            self.memory.insert(0, transition)
            self.memory = self.memory[:-1]
        else:
            self.memory.insert(0, transition)

    def sample(self, batch_size):
        # samples from memory
        if self.useTrick:
            # random batch from entire memory, reduces i.i.d.
            return random.sample(self.memory, batch_size)
        else:
            # latest batch_size memories, very correlated
            return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    # calculate epsilon for number of iteration
    # starts at 1, ends at 0.05, anneals over 1000 steps
    return max(1 - 0.00095*it, 0.05)

def select_action(model, state, epsilon):
    # Select action given the model, the state, epsilon, and the output from the Q-function
    _rand = random.random()
    state = torch.from_numpy(state).float()
    if _rand < epsilon:
        # take random action with epsilon probability
        return torch.randint(2, (1,)).item()
    else:
        with torch.no_grad():
            # take best action with 1-epsilon probability, actions decided by the Q-function (neural network)
            return torch.argmax(model(state)).item()

s = env.reset()
a = select_action(model, s, 0.05)
assert not torch.is_tensor(a)

def compute_q_val(model, state, action):
    # computes q-values
    # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    # model(state) contains estimated q values from the model
    # action contains the action chosen by the model at that point
    return model(state).gather(1, action.unsqueeze(1))
    
    
class targetComputer():
    """
    Computes the target to compute the loss, also keeps track of the fixed target network

    IN during initialization: 
    target_network_steps: 1 being no fixed network, anything >1 meaning keeping the target network fixed for so many steps
    """
    def __init__(self, target_network_steps=1):
        self.UPDATE_STEPS = target_network_steps
        self.target_network_steps = self.UPDATE_STEPS
        
    def compute_target(self, model, reward, next_state, done, discount_factor):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)

        # only update self.model to given model if target_network_steps = 0
        if not hasattr(self, 'model'):
            self.model = copy.deepcopy(model)
        elif self.target_network_steps == 0:
            self.model = model
            self.target_network_steps = self.UPDATE_STEPS
        
        max_Q_prime = torch.zeros(next_state.shape[0])  # batch size

        # create mask of non-done states
        non_final_mask = torch.tensor(tuple(map(lambda s: s != 1, done)), dtype=torch.bool)

        # when state is not done we add discount_factor * max() else 0
        max_Q_prime[non_final_mask] = torch.max(self.model(next_state[non_final_mask]), 1).values
        
        self.target_network_steps -= 1

        return (reward + discount_factor * max_Q_prime).unsqueeze(1)

def train(model, memory, optimizer, batch_size, discount_factor, TargetComputer, train=True):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None, 0

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean
    
    # compute the q value
    q_val = compute_q_val(model, state, action)
    
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = TargetComputer.compute_target(model, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to t Network (PyTorch magic)
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item(), q_val.max().item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, TargetComputer):
    
    optimizer = optim.Adam(model.parameters(), learn_rate)

    EVAL_EPS = 0.05 #epsilon during evaluation
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = [] 
    q_values = []     # used to experiment soft divergence
    max_q_values = [] # used to experiment soft divergence
    losses = []
    
    
    for i in range(num_episodes):
        # following algo from here: https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view
        episode_duration = 0
        s = env.reset()
        while True:
            a = select_action(model, s, get_epsilon(global_steps)) # epsilon-greedy select action
            s_next, r, done, _ = env.step(a)  # execute action a_t in emulator
            episode_duration += 1
            memory.push((s, a, r, s_next, done))  # store transition in memory
            global_steps += 1
            s = s_next
            if done:
                break

            # this is actually training as I do not pass it "train=False"
            loss, q_val = train(model, memory, optimizer, batch_size, discount_factor, TargetComputer)

            # only tracking losses from the evaluation now
            # losses.append(loss)

        # add  evaluation every 5th run with same epsilon
        if i % 5 == 0:
            episode_duration = 0
            s = env.reset()
            while True:
                a = select_action(model, s, EVAL_EPS) # epsilon-greedy select action
                s_next, r, done, _ = env.step(a)  # execute action a_t in emulator
                episode_duration += 1
                memory.push((s, a, r, s_next, done))  # store transition in memory
                global_steps += 1
                s = s_next
                if done:
                    break

                # This is NOT training, as I pass train=False
                loss, q_val = train(model, memory, optimizer, batch_size, discount_factor, TargetComputer, train=False)
                losses.append(loss)
                q_values.append(q_val)
            episode_durations.append(episode_duration)
            max_q_values.append(max(q_values))
    return episode_durations, max_q_values


def get_state_number(env):
    # A function that can be used to easily run several environments, get the number of states to pass to the Q-function
    if isinstance(env.env.observation_space, gym.spaces.box.Box):
        nS = env.env.observation_space.shape[0]
    elif isinstance(env.env.observation_space, gym.spaces.discrete.Discrete):
        nS = env.env.observation_space.n
    else:
        print("Encountered unknown class type in state: {}. Exiting code execution".format(type(item[1])))
        exit()    
    return nS

def get_action_number(env):
    # A function that can be used to easily run several environments, get the number of actions to pass to the Q-function
    if isinstance(env.env.action_space, gym.spaces.box.Box):
        nA = env.env.action_space.shape[0]
    elif isinstance(env.env.action_space, gym.spaces.discrete.Discrete):
        nA = env.env.action_space.n
    else:
        print("Encountered unknown class type in action: {}. Exiting code execution".format(type(item[1])))
        exit()           
    return nA

def run_experiment(hyperparams, _seed):
    num_episodes = hyperparams['num_episodes']
    batch_size = hyperparams['batch_size']
    discount_factor = hyperparams['discount_factor']
    learn_rate = hyperparams['learn_rate']
    memory = hyperparams['memory']
    num_hidden = hyperparams['num_hidden']
    seed = hyperparams['seed']
    target_network_steps = hyperparams['target_network_steps']
    env = hyperparams['env']

    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    random.seed(_seed)
    torch.manual_seed(_seed)
    env.seed(_seed)
    
    nA = get_action_number(env)
    nS = get_state_number(env)

    model = QNetwork(num_hidden=num_hidden, num_a=nA, num_s=nS)

    TargetComputer = targetComputer(target_network_steps=target_network_steps)

    episode_durations, max_q_values = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, TargetComputer)

    # And see the results
    return episode_durations, max_q_values



mountainEnv = gym.envs.make("MountainCar-v0")
acrobotEnv = gym.envs.make("Acrobot-v1")
cartPoleEnv = gym.envs.make("CartPole-v1")

envs = [mountainEnv, acrobotEnv, cartPoleEnv] # could run for these environments

hyperparams = {
    'env':                  cartPoleEnv,  # the environment is chosen to be cartpole
    'num_episodes':         100,  
    'batch_size':           64,
    'discount_factor':      0.8,   
    'learn_rate':           1e-3,
    'memory':               ReplayMemory(10000, useTrick=True), # set useTrick=False for no memory replay
    'num_hidden':           128,
    'seed':                 10, 
    'target_network_steps': 1 
}

repeats = 10
_seed = 0

results={}
q_results={}

for replay in [True, False]:

    hyperparams['memory'] = ReplayMemory(10000, useTrick=replay)
    
    for steps in [1, 50, 100, 200, 500]:
    
        results['episode'] = []
        q_results['episode'] = []
        
        hyperparams['target_network_steps'] = steps
        
        key = 'replay' if replay else 'no-replay'
        
        identifier =  '{}_steps{}_repeats{}'.format(key, steps, repeats)
        q_identifier = 'max_q_{}_steps{}_repeats{}'.format(key, steps, repeats)

        print('Running {}'.format(identifier))
        
        results[identifier] = []
        q_results[q_identifier] = []
        
        for repeat in range(repeats):
            _seed += 1                      # changing seed
            num_episodes, max_q_values = run_experiment(hyperparams, _seed)
            results[identifier].extend(num_episodes)
            q_results[q_identifier].extend(max_q_values)
            results['episode'].extend(list(range(hyperparams['num_episodes'])[::5])) #added :5 here, which is the number of evaluation steps we do
            q_results['episode'].extend(list(range(hyperparams['num_episodes'])[::5]))
            
with open('results_{}.pkl'.format(int(time.time())), 'wb') as f:
    pickle.dump(results, f)
    
results_df = pd.DataFrame(results)
q_results_df = pd.DataFrame(q_results)

# pd.melt is required to get several plots in one graph, got it from 
# https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn 2nd answer

plt.figure()
lineplot = sns.lineplot(x='episode', y='value', hue='variable', 
             data=pd.melt(results_df, ['episode']), ci=95)
fig = lineplot.get_figure()
fig.savefig("full-fig-{}.png".format(int(time.time())))
