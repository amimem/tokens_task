import gym
import gym_tokens
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import utils
import datetime
import sys

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

torch.manual_seed(0)

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))
env = gym.make("tokens-v0", alpha=0.75, seed=0, terminal=11, fancy_discount=False, v="terminate").unwrapped

def _mapFromIndexToTrueActions(actions):
	if actions == 1:
		return -1 
	elif actions == 2:
		return 1
	else:
		return 0

def _augState(stateVal, height = 11):
	"""
	Eg. Augment state value so that [-15,15] goes to [0,30] 
	"""
	return stateVal + height

def _sign(num):

	if num < 0:
		return -1

	elif num > 0:
		return 1

	else:
		return 0

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])

def get_screen():
	# Returned screen requested by gym is 400x600x3, but is sometimes larger
	# such as 800x1200x3. Transpose it into torch order (CHW).
	screen = env.render(mode='rgb_array').transpose((2, 0, 1))
	# Cart is in the lower half, so strip off the top and bottom of the screen
	_, screen_height, screen_width = screen.shape
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	# env.close()
	return resize(screen).unsqueeze(0).to(device)

#create train dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{'Tokens0'}_{'reinforce'}_seed{'0'}_{date}"

model_name = default_model_name
model_dir = utils.get_model_dir(model_name)

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
# txt_logger.info("{}\n".format(args))

if __name__ == "__main__":

	height = 11
	num_episode = 0
	episode_returns = []
	numRecentCorrectChoice = []
	totalReturns = [] # Return per episode
	choice_made = []
	correct_choice = []
	finalDecisionTime = []
	finalRewardPerGame = []
	traj_group = []
	env_name = 'tokens-v0'
	numCorrectChoice = 0
	last_choice = 0
	decisionTime = np.zeros(shape=((height*2)+1)) # histogram of decision times per episode

	# set up matplotlib
	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display

	plt.ion()

	# if gpu is to be used
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	BATCH_SIZE = 128
	GAMMA = 0.8
	EPS_START = 0.01
	EPS_END = 0.0001
	EPS_DECAY = 10000
	TARGET_UPDATE = 10

	# Get screen size so that we can initialize layers correctly based on shape
	# returned from AI gym. Typical dimensions at this point are close to 3x40x90
	# which is the result of a clamped and down-scaled render buffer in get_screen()
	init_screen = get_screen()
	_, _, screen_height, screen_width = init_screen.shape
	print(init_screen.shape)

	# Get number of actions from gym action space
	n_actions = env.action_space.n

	policy_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(10000)


	steps_done = 0


	def select_action(state):
		global steps_done
		sample = random.random()
		eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
		steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
				# t.max(1) will return largest column value of each row.
				# second column on max result is index of where max element was
				# found, so we pick action with the larger expected reward.
				return policy_net(state).max(1)[1].view(1, 1)
		else:
			return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


	episode_durations = []


	# def plot_durations():
	# 	plt.figure(2)
	# 	plt.clf()
	# 	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	# 	plt.title('Training...')
	# 	plt.xlabel('Episode')
	# 	plt.ylabel('Duration')
	# 	plt.plot(durations_t.numpy())
	# 	# Take 100 episode averages and plot them too
	# 	if len(durations_t) >= 100:
	# 		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
	# 		means = torch.cat((torch.zeros(99), means))
	# 		plt.plot(means.numpy())

	# 	plt.pause(0.001)  # pause a bit so that plots are updated
	# 	if is_ipython:
	# 		display.clear_output(wait=True)
	# 		display.display(plt.gcf())


	def optimize_model():
		if len(memory) < BATCH_SIZE:
			return
		transitions = memory.sample(BATCH_SIZE)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
		#FIXME:
		# The batch next state is all none, how to remove none?
		# Test it in the classical control env.
													if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(BATCH_SIZE, device=device)
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * GAMMA) + reward_batch

		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for param in policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()

	num_episodes = 10000
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		env.reset()
		last_screen = get_screen()
		current_screen = get_screen()
		state = current_screen - last_screen

		for t in count():
			# Select and perform an action
			action = select_action(state)
			nstate, reward, done, _ = env.step(_mapFromIndexToTrueActions(action.item()))
			reward = torch.tensor([reward], device=device)

			# Observe new state
			last_screen = current_screen
			current_screen = get_screen()
			if not done:
				next_state = current_screen - last_screen
			else:
				next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			if not done:
				# Move to the next state
				state = next_state

			# Perform one step of the optimization (on the target network)
			optimize_model()
			if done:
				num_episode += 1
				traj = env.get_trajectory()
				episode_durations.append(t + 1)
				totalReturns.append(reward) # reward per episode
				traj_group.append(traj)

				if reward > 0:
					numCorrectChoice += 1
					numRecentCorrectChoice.append(1)
				else:
					numRecentCorrectChoice.append(0) # binary value, correct choice or not per episode

				decision_step = _augState(abs(nstate[1])) # taking abs means that decision step is always between 15 and 31
				decisionTime[decision_step-1] += 1 # after each episode is done, one is added to the corresponding element in decision time,
				# so after 100 episodes, we have a histogram of decision times

				if abs(nstate[1]) == 0: # if we made no decision till the end
					last_choice += 1 # last choice represents the number of episodes in which we waited until the end
				
				choice_made.append(_sign(nstate[1])) # these arays are updated after each episode, not after each timestep
				correct_choice.append(_sign(traj[-1]))
				finalDecisionTime.append(abs(nstate[1])) # Why next_state? because it is the latest state that we have and we don't update state until after the if-else condition
				if env_name == 'tokens-v3' or env_name == 'tokens-v4':
					finalRewardPerGame.append(env.reward)
				else:
					finalRewardPerGame.append(reward.item())
				# plot_durations()
				break

		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())

		# if num_episodes % 10 == 0: # if the game has not stpped and we moved an episode forward

		# duration = int(time.time() - start_time)
		totalReturn_val = np.sum(totalReturns) # sum of all episodic returns

		avg_returns = np.mean(totalReturns[-1000:])
		recent_correct = np.mean(numRecentCorrectChoice[-1000:])

		header = ["Game"]
		data = [num_episode] # update and num_frames are +=15 ed

		header += ["Returns", "Avg Returns", "Correct Percentage", "Recent Correct", "decision_time"]
		data += [totalReturn_val.item(), avg_returns.item(), numCorrectChoice/num_episode, recent_correct, finalDecisionTime[num_episode-1]]

		txt_logger.info(
			"G {} | R {:.3f} | Avg R {:.3f} | Avg C {:.3f} | Rec C {:.3f} | DT {}"
			.format(*data))

		csv_header = ["trajectory", "choice_made", "correct_choice", "decision_time", "reward_received"]
		csv_data = [traj_group[num_episode-1], choice_made[num_episode-1], correct_choice[num_episode-1], finalDecisionTime[num_episode-1], finalRewardPerGame[num_episode-1]]

		if num_episode == 1:
			csv_logger.writerow(csv_header)
		csv_logger.writerow(csv_data)
		csv_file.flush()

		# Save status
		if num_episode % 100 == 0:
			# status = {"num_frames": num_frames, "update": update, "games": num_games, "totalReturns" : totalReturns}
			# model.save_q_state(model_dir, num_games)
			np.save(model_dir+'/decisionTime_'+str(num_episode)+'.npy', decisionTime)
			# txt_logger.info("Status saved")
			# utils.save_status(status, model_dir)

	print('Complete')
	env.render()
	env.close()
	plt.ioff()
	plt.show()
	plt.savefig("plotdur.png")