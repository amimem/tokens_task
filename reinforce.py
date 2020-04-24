"""
Taken from Pytorch examples
"""

import argparse
import gym
import gym_tokens
import numpy as np
from itertools import count
import random
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

gamma = 0.8
seed = 0
lr= 0.001
render = False
log_interval = 1

utils.seed(seed)

env = gym.make('tokens-v1', gamma=0.75, seed=seed, terminal=15, fancy_discount=False)
env.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def _mapFromIndexToTrueActions(actions):
	if actions == 1:
		return -1 
	elif actions == 2:
		return 1
	else:
		return 0

in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n
h_dim_p = 128
class Policy(nn.Module):
	def __init__(self, in_dim, h_dim, out_dim):
		super(Policy, self).__init__()
		self.linear_1 = nn.Linear(in_dim, h_dim, bias=True)
		self.relu_1 = nn.ReLU()
		self.linear_2 = nn.Linear(h_dim, out_dim, bias=True)
		self.softmax = nn.Softmax(dim=1)

		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		o_1 = self.linear_1(x)
		o_2 = self.relu_1(o_1)
		o_3 = self.linear_2(o_2)
		o_4 = self.softmax(o_3)
		return o_4


policy = Policy(in_dim, h_dim_p, out_dim)
# optimizer = optim.SGD(params=policy.parameters(), lr=lr)
# optimizer = optim.RMSprop(policy.parameters(), lr=lr)
optimizer = optim.Adam(policy.parameters(), lr=0.001)
eps = np.finfo(np.float32).eps.item()


def select_action(state, time_step):
	state = torch.from_numpy(state).float().unsqueeze(0)
	probs = policy(state)
	m = Categorical(probs)
	action = m.sample()
	s = state.numpy()
	# if action == 0 and time_step == 15 and s[0,1] == 0:
	# 	return random.choice([-1,1])
	# else:	
	policy.saved_log_probs.append(m.log_prob(action))
	return _mapFromIndexToTrueActions(action.item())

	"""NOTE if we do a random choice, the length of the list of log prob = the length of rewards - 1
	Therefore, the reward, be it 1 or -1 or 0, would not have any effect on the weights.
	So if we waited until the end and took no action before, policy remains unchanged ! 
	Now is it dangerous? Nope, because it is not reinforced, and there will be a trajectory in which taking the right action will reinforce good behaviour.
	Cases in which the agent receives a reward if no random action is chosen:
		sign(0 = coordinate), sign(0 == action)
	Cases in which the agent receives a reward if a random action is chosen:
	sign(0 = coordinate), sign(0 == action)
	sign(+ = coordinate), sign(1 == action)
	sign(- = coordinate), sign(-1 == action)
	so choosing a random action is more rewarding in general !
	"""
	#NOTE The above note does not apply to the env

def finish_episode():
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + gamma * R
		returns.insert(0, R)
	returns = torch.tensor(returns).float()
	# returns = (returns - returns.mean()) / (returns.std() + eps)
	i = 0
	for log_prob, R in zip(policy.saved_log_probs, returns):
		policy_loss.append(-(gamma**i)*log_prob * R)
		i += 1
	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]


def _sign(num):
	'''
	This function determines the sign of the input value.
	: param num (int) : input
	: return num (int) : sign of input
	'''
	assert isinstance(num, np.int64)

	if num < 0:
		return -1

	elif num > 0:
		return 1

	else:
		return 0

returns = []
num_correct= 0
for i_episode in range(1,10001):
	state, ts = env.reset()
	ep_reward = 0
	done = False

	time_step = 0
	while not done:  # Don't infinite loop while learning
		action = select_action(state, time_step)

		#FIXME Actions are chosen even if they cause no effect on state.
		#their log prob is used to adjust the weight

		state, reward, done, ts = env.step(action)
		time_step += 1

		if render:
			env.render()
		policy.rewards.append(reward)
		ep_reward += reward
		if done:
			returns.append(ep_reward)
			if (_sign(state[1]) == _sign(state[0])):
				num_correct += 1
			break

	finish_episode()
	if i_episode % log_interval == 0:
		print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.3f}\tRec reward: {:.3f}\tAvg Correct: {:.3f}'.format(
				i_episode, ep_reward, np.mean(returns), np.mean(returns[-1000:]), num_correct/i_episode))