import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import unittest

class TokensEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, seed=7, terminal=3):

		np.random.seed(seed)

		self.num_actions = 3
		# forward or backward in each dimension
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(low=np.array([-15, -15]), high=np.array([15, 15]), dtype=np.int64)
		
		# initial condition
		self.state = np.zeros(2) #index 0: Nt, index 1: ht
		self.reset()
		self.terminal = terminal
		

	def step(self, action):
		'''
		# action = 0 as do nothing
		# action = -1, left
		# action = 1, right
		'''

		if action >1 or action <-1:
			raise Exception('action should belong to this set: [-1,0,1]')

		coinToss = np.random.uniform()
		Nt_prev = self.state[0]
		ht_prev = self.state[1]
		is_done = False

		#Go left if prob is less than 0.5, go right otw. 
		if coinToss <= 0.5:
			Nt = Nt_prev - 1

		else:
			Nt = Nt_prev + 1

		if ht_prev == 0:
			ht = ht_prev + (self.time_steps+1) * action

		else:
			ht = ht_prev

		if self.time_steps == self.terminal:
			reward = self._indicator(self._sign(Nt_prev),self._sign(ht))
			next_state = self.reset()
			is_done = True

			return next_state, reward, is_done, self.time_steps

		else:

			# coinToss = np.random.uniform()
			# Nt_prev = self.state[0]
			# ht_prev = self.state[1]
			# is_done = False

			# #Go left if prob is less than 0.5, go right otw. 
			# if coinToss <= 0.5:
			# 	Nt = Nt_prev - 1

			# else:
			# 	Nt = Nt_prev + 1

			# if ht_prev == 0:
			# 	ht = ht_prev + self.time_steps * action
			# else:
			# 	ht = ht_prev
			
			next_state = np.zeros(2,dtype=np.int64)
			next_state[0] = Nt
			next_state[1] = ht
			self.state = next_state

			reward = 0
			self.time_steps += 1


		# else:
		# 	reward = self._indicator(self._sign(Nt),self._sign(ht))
		# 	next_state = self.reset()
		# 	is_done = True

			

	
		return next_state, reward, is_done, self.time_steps

	def _sign(self, num):

		assert isinstance(num, np.int64)

		if num < 0:
			return -1

		elif num > 0:
			return 1

		else:
			return 0

	def _indicator(self, num1, num2):
		if num1 == num2:
			return 1
		else:
			return 0

	def get_num_states(self):
		return len(range(-15,16))*len(range(-15,16)) #-15 to 15 inclusive for Nt and -15 to 15 inclusive for ht

	def get_num_actions(self):
		return self.num_actions





	def reset(self):
		self.state = np.zeros(2, dtype=np.int64)
		self.done = False
		self.time_steps = 0

		return self.state