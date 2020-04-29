import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import unittest

class TokensEnvS(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, alpha, seed=7, terminal=3, fancy_discount=False, v='terminate'):
		'''
		This is the constructor for the tokens env.
		: param alpha (float): discount factor
		: param seed (int): random seed value
		: param terminal (int): max time step for the environment
		: param fancy_discount (boolean): uses fancy discounting to compute reward 
		'''

		np.random.seed(seed)

		self.num_actions = 3
		# forward or backward in each dimension
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(low=np.array([-terminal, -terminal, 0]), high=np.array([terminal, terminal, terminal]), dtype=np.int64)

		
		# initial condition
		self.state = np.zeros(3) #index 0: Nt, index 1: ht, index 2: time_step
		self.alpha = alpha
		self.terminal = terminal
		self.fancy_discount = fancy_discount
		self.trajectory = [0]
		self.v = v
		self.reset()

	def step(self, action):
		'''
		The function takes in action and send that action to the environment. 
		: param action :(integer consisting of [-1,0,1])
		: return next state, reward, is_done (boolean) and in-game time steps
		Representation used for actions:
			i) action = 0 as do nothing
			ii) action = -1, go left
			iii) action = 1, go right
		'''

		if action >1 or action <-1:
			raise Exception('action should belong to this set: [-1,0,1]')

		Nt_prev = self.state[0]
		ht_prev = self.state[1]
		is_done = False

		#Play action if all previous actions are waiting (action = 0). Else, preserve previous played actions
		if ht_prev == 0:
			ht = ht_prev + (self.time_steps+1) * action

		else:
			ht = ht_prev

		#If in-game time has reached max time step, assign a reward value if the correct side (based on sign) is chosen.

		Nt = Nt_prev
		next_state = np.zeros(3,dtype=np.int64)
		dec_time = self.time_steps

		if ht:
			
			next_state[0] = Nt # set n before
			# Nt was set before
			next_state[1] = ht
			is_done = True

			reward = self.get_pt_plus(next_state[0], dec_time)

			next_state[2] = dec_time
			self.state = next_state
			return next_state, reward, is_done, self.time_steps

		else:

			if self.time_steps < self.terminal:
				Nt = self.trajectory[self.time_steps + 1]
				self.time_steps += 1
				is_done = False
			else:
				is_done = True

			next_state[0] = Nt # set n after
			next_state[1] = ht
			reward = 0
			next_state[2] = self.time_steps
			self.state = next_state
			return next_state, reward, is_done, self.time_steps

		"""
		Here is what happens if step:
		if the action is taken, only ht , time_step is changed, reward is calculated and the new state is returned.
		This makes sense since for the Q table the state (N,0,t) and non zero action A leads to a good return.
		So the agent is motivated to take this action again when visiting the same state.
		q value of next state is 0 so (N, ht, t_dec) or any other state return does not have much effect.
		"""

	def _fancy_discount_reward(self, reward, inter_trial_interval = 7.5):
		'''
		This function computes fancy discounting
		: param reward (int) : reward value before any discounting is applied. Value should be 0 or 1. 
		: return (float) : fancy discounted reward
		'''
		return reward / self.terminal / (np.absolute(self.state[1])/self.terminal + self.alpha * (1 - np.absolute(self.state[1]) / self.terminal) + inter_trial_interval/self.terminal)

	def _sign(self, num):
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

	def _indicator(self, num1, num2):
		'''
		This function returns 1 if both inputs are the same value
		: param num1 : input 1
		: param num2 : input 2
		: return num (int) : 1 if both inputs are the same and 0 otherwise
		'''
		if num1 == num2:
			return 1
		else:
			return 0

	def get_num_states(self):
		'''
		This function computes the total number of states
		: return (int) : total number of states
		'''
		return len(range(-self.terminal,self.terminal+1))*len(range(-self.terminal,self.terminal+1))*len(range(self.terminal+1)) #-15 to 15 inclusive for Nt and -15 to 15 inclusive for ht, 0 to 15 inclusive for time_steps


	def get_num_actions(self):
		'''
		This function returns the number of available actions of the environment.
		: return (int) : number of actions
		'''
		return self.num_actions

	def get_trajectory(self):
		'''
		This function returns the number of available actions of the environment.
		: return (int) : number of actions
		'''
		return self.trajectory

	def reset(self):
		'''
		This function resets the environment by setting the states, time_steps to zero
		'''
		self.state = np.zeros(3, dtype=np.int64)
		self.done = False
		self.time_steps = 0
		self.trajectory = [0]
		self.set_trajectory()

		return self.state, self.time_steps 

	def set_trajectory(self):
		Nt = self.state[0]
		t = self.time_steps
		while t < self.terminal:

			if np.random.uniform() <= 0.5:
				Nt -= 1
			else:
				Nt += 1

			self.trajectory.append(Nt)
			t += 1
		
	def binomial(self, n, k):
		if 0 <= k <= n:
			ntok = 1
			ktok = 1
			for t in range(1, min(k, n - k) + 1):
				ntok *= n
				ktok *= t
				n -= 1
			return ntok // ktok
		else:
			return 0

	def get_pt_plus(self, Nt, t, p=0.5):
		'''
		Outputs probability that reward>=0 at time T if state is Nt at time t, for +ve jump probability, p.
		Behaves strangely for T a multiple of 10...
		'''
		T = self.terminal
		if t==-1:
			return p
		else:
			tp = T - t 
			Nt_plus=(t+Nt)/2.
			if tp<(T-1)/2.-(t-Nt)/2.:
				if Nt>0:
					return 1
				else:
					return 0
			else:
				NL=(t-Nt)/2.
				Nc=T-t
				kvec=np.arange(0,(np.min((Nc,(T-1)/2-NL))+1))
				return np.power(p,Nc)*np.sum(np.asarray([self.binomial(int(Nc), int(k)) for k in kvec]))
