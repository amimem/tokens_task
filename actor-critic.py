import gym
import gym_tokens
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.distributions import Categorical
import scipy.signal as signal

import time
import datetime
import sys
import utils
import lib

import numpy as np


class Actor(nn.Module):
	def __init__(self, in_dim, h_dim, out_dim_p):
		super(Actor, self).__init__()
		self.linear_1 = nn.Linear(in_dim, h_dim, bias=True)
		self.relu_1 = nn.ReLU()
		# self.linear_2 = nn.Linear(h_dim, 256, bias=True)
		# self.relu_2 = nn.ReLU()
		self.linear_p = nn.Linear(h_dim, out_dim_p, bias=True)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input):
		o_1 = self.linear_1(input)
		o_2 = self.relu_1(o_1)
		# o_3 = self.linear_2(o_2)
		# o_4 = self.relu_2(o_3)
		o_5 = self.linear_p(o_2)
		p = self.softmax(o_5)
		return p

class Critic(nn.Module):
	def __init__(self, in_dim, h_dim, out_dim_v):
		super(Critic, self).__init__()
		self.linear_1 = nn.Linear(in_dim, h_dim, bias=True)
		self.relu_1 = nn.ReLU()
		self.linear_v = nn.Linear(h_dim, out_dim_v, bias=True)

	def forward(self, input):
		o_1 = self.linear_1(input)
		o_2 = self.relu_1(o_1)
		v = self.linear_v(o_2)
		return v

def _sign(num):

	if num < 0:
		return -1

	elif num > 0:
		return 1

	else:
		return 0

def _mapFromIndexToTrueActions(actions):
	if actions == 1:
		return -1 
	elif actions == 2:
		return 1
	else:
		return 0

def _augState(stateVal, height):
	"""
	Eg. Augment state value so that [-15,15] goes to [0,30] 
	"""
	return stateVal + height

def a2c():

	parser = argparse.ArgumentParser()

	parser.add_argument("--games", type=int, default=100000, help="number of games training (default: 500)")
	parser.add_argument("--env",  default='tokens-v0' , help="name of the environment to train on (REQUIRED)")
	parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
	parser.add_argument("--log_interval", type=int, default=1, help="number of updates between two logs (default: 1)")
	parser.add_argument("--convg", type=float, default=0.00001, help="convergence value")
	parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
	parser.add_argument("--save-interval", type=int, default=20000, help="number of updates between two saves (default: 30, 0 means no saving)")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--height", type=int, default=15, help="game tree height")
	parser.add_argument('--fancy_discount', help='use fancy discounting rewards',action='store_true')
	parser.add_argument('--fast_block', help='fast block discounting',action='store_true')
	parser.add_argument('--variation', default="horizon", help='which variation')

	args = parser.parse_args()

	#create train dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"{args.env}_{'a2c'}_seed{args.seed}_{date}"

	model_name = default_model_name
	model_dir = utils.get_model_dir(model_name)

	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	if args.fast_block:
		block_discount = 0.25

	else:
		block_discount = 0.75

	env = gym.make(args.env, alpha=block_discount, seed=args.seed, terminal=args.height, fancy_discount=args.fancy_discount, v=args.variation)
	txt_logger.info("Environments loaded\n")

	status = {"num_episode":0}
	txt_logger.info("Training status loaded\n")

	# Set seed for all randomness sources
	utils.seed(args.seed)

	num_actions = env.get_num_actions()

	num_episode = status["num_episode"]
	prev_num_episode = 0

	start_time = time.time()

	episodes_decison_times = []
	episodes_loss = []

	episodes_decison_times = np.zeros(shape=((args.height*2)+1)) # histogram of decision times per episode
	#NOTE why first 15 elements are always empty

	episode_choice = [] # choise of each episode
	correct_choice = []
	final_episode_decison_time = []
	episode_returns = []
	numCorrectChoice = 0
	numRecentCorrectChoice = []

	lr = args.lr
	h_dim_p = 256
	h_dim_v = 64
	input_shape = env.observation_space.shape[0]

	last = 0
	# create the actor network and it's optimizer
	actor = Actor(env.observation_space.shape[0], h_dim_p ,env.action_space.n)
	optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)

	# create the critic network and it's optimizer
	critic = Critic(env.observation_space.shape[0], h_dim_v , 1)
	optimizer_critic = optim.Adam(critic.parameters(), lr=0.01)

	run_trajectories = []

	for n in range(args.games): # for each episode

		s, _ = env.reset() # reset the environment to get the initial state

		I = 1 # set I = 1 (psuedo-code of page 332 of Sutton's book)
		done = False

		while not done:

			# choose an action based on the actor policy
			p = actor(torch.from_numpy(s).unsqueeze(0).type(torch.FloatTensor))
			m = Categorical(p)
			a = m.sample()

			# take the action
			s_prime, reward, done, _ = env.step(_mapFromIndexToTrueActions(a.item()))

			if done: # if done , v_hat_s_prime = 0
			  v_hat_s_prime = 0
			else: # otherwise get its value form the critic network
			  v_hat_s_prime = critic(torch.from_numpy(s_prime).unsqueeze(0).type(torch.FloatTensor))

			v_hat_s = critic(torch.from_numpy(s).unsqueeze(0).type(torch.FloatTensor)) # get v_hat_s from the critic network

			delta = reward + args.gamma*v_hat_s_prime - v_hat_s # compute delta = r + gamma*v_hat(s',w) - v_hat(s,w)

			loss_v = delta.pow(2) # specify the loss for the value (critic) network

			# clear gradients, perform backward propagation for the critic network
			optimizer_critic.zero_grad()
			loss_v.backward()
			optimizer_critic.step()

			loss_p = -m.log_prob(a)*delta.detach()*I # specify the loss for the policy (actor) network, detaching delta is important

			# clear gradients, perform backward propagation for the policy network
			optimizer_actor.zero_grad()
			loss_p.backward()
			optimizer_actor.step()

			# update I and s
			I *= args.gamma
			s = s_prime

		num_episode+=1
		run_trajectories.append(env.get_trajectory())
		decision_step = _augState(abs(s[1]), args.height) # taking abs means that decision step is always between 15 and 31
		episodes_decison_times[decision_step-1] += 1 # after each episode is done, one is added to the corresponding element in decision time,
		#FIXME check the index
		# so after 100 episodes, we have a histogram of decision times

		if abs(s[1]) == args.height+1: # if we made no decision till the end
			last += 1 # last choice represents the number of episodes in which we waited until the end

		episode_choice.append(_sign(s[1])) # these arays are updated after each episode, not after each timestep
		correct_choice.append(_sign(env.trajectory[-1]))
		final_episode_decison_time.append(abs(s[1])) # Why next_state? because it is the latest state that we have and we don't update state until after the if-else condition

		if args.env == 'tokens-v3' or args.env == 'tokens-v4':
			episode_returns.append(env.reward) # reward per episode
			if env.reward > 0:
				numCorrectChoice += 1
				numRecentCorrectChoice.append(1)
			else:
				numRecentCorrectChoice.append(0) # binary value, correct choice or not per episode

		else:
			episode_returns.append(reward) # reward per episode
			if reward > 0:
				numCorrectChoice += 1
				numRecentCorrectChoice.append(1)
			else:
				numRecentCorrectChoice.append(0) # binary value, correct choice or not per episode

		if num_episode > prev_num_episode and num_episode % args.log_interval == 0: # if the game has not stpped and we moved an episode forward

			duration = int(time.time() - start_time)
			episode_total_loss = np.sum(episodes_loss) # sum of all episodic losses
			totalReturn_val = np.sum(episode_returns) # sum of all episodic returns

			avg_loss = np.mean(episodes_loss[-1000:])
			avg_returns = np.mean(episode_returns[-1000:])
			recent_correct = np.mean(numRecentCorrectChoice[-1000:])

			header = ["Games", "duration"]
			data = [num_episode, duration] # update and num_frames are +=15 ed

			header += ["lr", "last"]
			data += [lr, last]

			header += ["Returns", "Avg Returns", "Correct Percentage", "Recent Correct", "decision_time"]
			data += [totalReturn_val.item(), avg_returns.item(), numCorrectChoice/num_episode, recent_correct, final_episode_decison_time[prev_num_episode]]

			txt_logger.info(
				"G {} | D {} | LR {:.5f} | Last {} | R {:.3f} | Avg R {:.3f} | Avg C {:.3f} | Rec C {:.3f} | DT {}"
				.format(*data))

			csv_header = ["trajectory", "choice_made", "correct_choice", "decision_time", "reward_received"]
			csv_data = [run_trajectories[prev_num_episode], episode_choice[prev_num_episode], correct_choice[prev_num_episode], final_episode_decison_time[prev_num_episode], episode_returns[prev_num_episode]]

			if num_episode == 1:
				csv_logger.writerow(csv_header)
			csv_logger.writerow(csv_data)
			csv_file.flush()
			
			prev_num_episode = num_episode

if __name__ == "__main__":
	a2c()
