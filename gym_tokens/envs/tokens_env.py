import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TokensEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):

	# forward or backward in each dimension
	self.action_space = spaces.Discrete(3)
	self.observation_space = spaces.Box(low=(-15,-15), high=(15,15), size=(2), dtype=np.int64)
	
	# initial condition
	self.state = np.zeros(2) #index 0: Nt, index 1: ht
	self.reset()
	self.seed()

	def step(self, action):
		'''
		action = 0 as do nothing
		action = 1, convert to left (-1)
		action = 2, convert to right (+1)
		'''

		self.time_steps += 1

		coinToss = np.random.uniform()
		Nt_prev = self.state[0]
		ht_prev = self.state[1]
		is_done = False

		if action == 1 and not is_done:
			mod_action = -1

		elif action == 2 and not is_done:
			mod_action = 1

		else:
			mod_action = 0



		#Go left if prob is less than 0.5, go right otw. 
		if coinToss <= 0.5:
			Nt = Nt_prev - 1

		else:
			Nt = Nt_prev + 1

		if action != 0 and not self.flip_H:
			ht = ht_prev + self.time_steps * mod_action
			self.flip_H = True

		elif self.flip_H:
			ht = ht_prev + self.time_steps * mod_action

		next_state = np.zeros(2)
		next_state[0] = Nt
		next_state[1] = ht

		if self.time_steps < 15:
			reward = 0

		else:
			reward = np.sign([Nt]) * np.sign([ht])
			self.reset()
			is_done = True

		self.state = next_state


		return next_state, action, is_done, _





	def reset(self):
		self.state = np.zeros(2)
		self.done = False
		self.time_steps = 0
		self.flip_H = False


	 def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]