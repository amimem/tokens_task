import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import unittest
import time
from gym.envs.classic_control import rendering

class TokensEnv(gym.Env):

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 50
		}

	def __init__(self, alpha, seed=7, terminal=3, fancy_discount=False, negative_reward=False, v='terminate'):
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
		self.reward = 1
		self.terminal = terminal
		self.counter = np.array([0, self.terminal, 0])
		self.reset()
		self.fancy_discount = fancy_discount
		self.trajectory = [0]
		self.v = v
		self.viewer = None
		self.negative_reward = negative_reward

	def step(self, action):
		if self.v == 'terminate':
			return self._step_v_terminate(action)
		elif self.v == 'horizon':
			return self._step_v_horizon(action)

	def _step_v_terminate(self, action):
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

			while self.time_steps < self.terminal:

				if np.random.uniform() <= 0.5:
					Nt -= 1
					# self.counter[0] += 1
				else:
					Nt += 1
					# self.counter[2] += 1

				self.trajectory.append(Nt)
				self.time_steps += 1

			is_done = True

			# Nt was set before
			next_state[1] = ht

			if self.fancy_discount:
				reward = self._indicator(self._sign(Nt),self._sign(ht))
				reward = self._fancy_discount_reward(reward)
			else:
				reward = self._indicator(self._sign(Nt),self._sign(ht))

			next_state[2] = dec_time
			self.state = next_state
			return next_state, reward, is_done, self.time_steps

		else:

			if self.time_steps < self.terminal:
				if np.random.uniform() <= 0.5:
					Nt -= 1
					self.counter[0] += 1
				else:
					Nt += 1
					self.counter[2] += 1

				self.trajectory.append(Nt)
				self.time_steps += 1
				is_done = False
			else:
				is_done = True

			next_state[0] = Nt # set n after
			next_state[1] = ht
			reward = 0
			next_state[2] = self.time_steps
			self.state = next_state
			self.counter[1] = self.terminal - self.counter[0] - self.counter[2]
			return next_state, reward, is_done, self.time_steps

		"""
		Here is what happens if step:
		if the action is taken, only ht , time_step is changed, reward is calculated and the new state is returned.
		This makes sense since for the Q table the state (N,0,t) and non zero action A leads to a good return.
		So the agent is motivated to take this action again when visiting the same state.
		q value of next state is 0 so (N, ht, t_dec) or any other state return does not have much effect.
		"""

	def _step_v_horizon(self, action):
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

		#Go left if prob is less than 0.5, go right otherwise if in-game time-steps less than max time-step
		if self.time_steps < self.terminal:

			if np.random.uniform() <= 0.5:
				Nt = Nt_prev - 1

			else:
				Nt = Nt_prev + 1

			self.trajectory.append(Nt)

		#When max time-step is reached, ensure that the final state observed (Nt) is the same as the previous
		else:
			Nt = Nt_prev

		#Play action if all previous actions are waiting (action = 0). Else, preserve previous played actions
		if ht_prev == 0:
			ht = ht_prev + (self.time_steps+1) * action

		else:
			ht = ht_prev


		#If in-game time has reached max time step, assign a reward value if the correct side (based on sign) is chosen.
		if self.time_steps == self.terminal:
			if ht == 0:
				reward = 0
			else:
				reward = self._indicator(self._sign(Nt),self._sign(ht))
			next_state = np.zeros(3,dtype=np.int64)
			next_state[0] = Nt
			next_state[1] = ht
			is_done = True

			#fancy discounting reward is applied if initialised when the environment is constructed
			if self.fancy_discount:
				if ht == 0:
					reward = 0
				else:
					reward = self._fancy_discount_reward(reward)

		else:

			next_state = np.zeros(3,dtype=np.int64)
			next_state[0] = Nt
			next_state[1] = ht
			self.state = next_state

			reward = 0
			self.time_steps += 1

		next_state[2] = self.time_steps
		return next_state, reward, is_done, self.time_steps

	def _fancy_discount_reward(self, reward, inter_trial_interval = 7.5):
		'''
		This function computes fancy discounting
		: param reward (int) : reward value before any discounting is applied. Value should be 0 or 1. 
		: return (float) : fancy discounted reward
		'''
		inter_trial_interval = self.terminal / 2.0
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
			return self.reward
		else:
			if self.negative_reward:
				return -(self.reward)
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

	def set_reward(self, reward):
		self.reward = reward

		# Taken from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
	def render(self, mode='human', state = None):
		# time.sleep(2)
		screen_width = 800
		screen_height = 300
		radius = 100
		radius_scale = 1.1
		token_radius = 8
		distance_scale = 2.5
		tokens_distance_scale = 3
		num_tokens = self.terminal

		if self.viewer is None:

			self.num_range = np.arange(start = -radius + 2*token_radius, stop = radius - token_radius , step = token_radius*tokens_distance_scale)
			self.coords_list = []

			while len(self.coords_list) < num_tokens:
				coords = np.random.choice(self.num_range, 2)

				while coords[0]**2 + coords[1]**2 > radius**2:
					coords = np.random.choice(self.num_range, 2)
				
				self.coords_list.append(coords)
				self.coords_list = [list(x) for x in {(tuple(e)) for e in self.coords_list}]

			self.token_translates = []

			for coords in self.coords_list:
				self.token_translates.append(rendering.Transform(translation = (coords[0], coords[1])))

			self.viewer = rendering.Viewer(screen_width, screen_height) # Creates a view using the specified width and height

			self.middle_trans = rendering.Transform(translation = (screen_width/2, screen_height/2))
			self.right_trans = rendering.Transform(translation = (distance_scale*radius, 0.0))
			self.left_trans = rendering.Transform(translation = (-distance_scale*radius, 0.0))

			self.middle_circle = rendering.make_circle(radius_scale*radius, 100, filled=False)
			self.middle_circle.add_attr(self.middle_trans)
			self.viewer.add_geom(self.middle_circle)

			self.right_circle = rendering.make_circle(radius_scale*radius, 100 , filled=False)
			self.right_circle.add_attr(self.middle_trans)
			self.right_circle.add_attr(self.right_trans)
			self.viewer.add_geom(self.right_circle)

			self.left_circle = rendering.make_circle(radius_scale*radius, 100 , filled=False)
			self.left_circle.add_attr(self.middle_trans)
			self.left_circle.add_attr(self.left_trans)
			self.viewer.add_geom(self.left_circle)

			for i in range(num_tokens):
				exec(f'self.token_{i} = rendering.make_circle(token_radius, filled=True)')
				exec(f'self.token_{i}.set_color(.0, .0, .0)')
				exec(f'self.token_{i}.add_attr(self.middle_trans)')
				exec(f'self.token_{i}.add_attr(self.token_translates[{i}])')
				exec(f'self.viewer.add_geom(self.token_{i})')

		if self.state is None:
			return None

		for i in range(self.counter[0]):
			exec(f'self.token_translates[{i}].set_translation(-distance_scale*radius + self.coords_list[{i}][0] ,  self.coords_list[{i}][1])')

		for i in range(self.counter[0], self.counter[2] + self.counter[0]):
			exec(f'self.token_translates[{i}].set_translation(distance_scale*radius + self.coords_list[{i}][0] , self.coords_list[{i}][1])')

		for i in range(self.counter[2] + self.counter[0] , np.sum(self.counter)):
			pass
			

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		else:
			print("Viewer is none")

	def reset(self):
		'''
		This function resets the environment by setting the states, time_steps to zero
		'''
		self.state = np.zeros(3, dtype=np.int64)
		self.counter = np.array([0, self.terminal, 0])
		self.done = False
		self.time_steps = 0
		self.trajectory = [0]

		return self.state, self.time_steps 


class TokensEnv2(gym.Env):
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
		self.observation_space = spaces.Box(low=np.array([-terminal+1, -terminal+1]), high=np.array([terminal+1, terminal+1]), dtype=np.int64)
		
		# initial condition
		self.state = np.zeros(2) #index 0: Nt, index 1: ht
		self.alpha = alpha
		self.reset()
		self.terminal = terminal
		self.fancy_discount = fancy_discount
		self.trajectory = [0]
		self.v = v

	def step(self, action):

		if self.v == 'terminate':
			return self._step_v_terminate(action)
		elif self.v == 'horizon':
			return self._step_v_horizon(action)

	def _step_v_terminate(self, action):	
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
		next_state = np.zeros(2,dtype=np.int64)
		dec_time = self.time_steps

		if ht:
			
			next_state[0] = Nt # set n before

			while self.time_steps < self.terminal:

				if np.random.uniform() <= 0.5:
					Nt -= 1
				else:
					Nt += 1

				self.trajectory.append(Nt)
				self.time_steps += 1

			is_done = True

			# Nt was set before
			next_state[1] = ht

			if self.fancy_discount:
				reward = self._indicator(self._sign(Nt),self._sign(ht))
				reward = self._fancy_discount_reward(reward)
			else:
				reward = self._indicator(self._sign(Nt),self._sign(ht))

			self.state = next_state
			return next_state, reward, is_done, self.time_steps

		else:

			if self.time_steps < self.terminal:
				if np.random.uniform() <= 0.5:
					Nt -= 1
				else:
					Nt += 1

				self.trajectory.append(Nt)
				self.time_steps += 1
				is_done = False
			else:
				is_done = True

			next_state[0] = Nt # set n after
			next_state[1] = ht
			reward = 0
			self.state = next_state
			return next_state, reward, is_done, self.time_steps

		"""
		Here is what happens if step:
		if the action is taken, only ht , time_step is changed, reward is calculated and the new state is returned.
		This makes sense since for the Q table the state (N,0,t) and non zero action A leads to a good return.
		So the agent is motivated to take this action again when visiting the same state.
		q value of next state is 0 so (N, ht, t_dec) or any other state return does not have much effect.
		"""

	def _step_v_horizon(self, action):
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

		#Go left if prob is less than 0.5, go right otherwise if in-game time-steps less than max time-step
		if self.time_steps < self.terminal:

			if np.random.uniform() <= 0.5:
				Nt = Nt_prev - 1

			else:
				Nt = Nt_prev + 1
			
			self.trajectory.append(Nt)

		#When max time-step is reached, ensure that the final state observed (Nt) is the same as the previous
		else:
			Nt = Nt_prev

		#Play action if all previous actions are waiting (action = 0). Else, preserve previous played actions
		if ht_prev == 0:
			ht = ht_prev + (self.time_steps+1) * action

		else:
			ht = ht_prev


		#If in-game time has reached max time step, assign a reward value if the correct side (based on sign) is chosen.
		if self.time_steps == self.terminal:
			if ht == 0:
				reward = 0
			else:
				reward = self._indicator(self._sign(Nt),self._sign(ht))
			next_state = np.zeros(2,dtype=np.int64)
			next_state[0] = Nt
			next_state[1] = ht
			is_done = True

			#fancy discounting reward is applied if initialised when the environment is constructed
			if self.fancy_discount:
				if ht == 0:
					reward = 0
				else:
					reward = self._fancy_discount_reward(reward)

		else:

			next_state = np.zeros(2,dtype=np.int64)
			next_state[0] = Nt
			next_state[1] = ht
			self.state = next_state

			reward = 0
			self.time_steps += 1

		return next_state, reward, is_done, self.time_steps

	def _fancy_discount_reward(self, reward, inter_trial_interval = 7.5):
		'''
		This function computes fancy discounting
		: param reward (int) : reward value before any discounting is applied. Value should be 0 or 1. 
		: return (float) : fancy discounted reward
		'''
		inter_trial_interval = self.terminal / 2.0
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
		return len(range(-self.terminal,self.terminal+1))*len(range(-self.terminal,self.terminal+1)) #-15 to 15 inclusive for Nt and -15 to 15 inclusive for ht

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

	# Taken from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height) # Creates a view using the specified width and height

			self.trans = rendering.Transform()

			self.axle = rendering.make_circle(50, filled=False)
			self.axle.add_attr(self.trans)
			self.axle.set_color(.5, .5, .8)
			self.viewer.add_geom(self.axle)

		if self.state is None:
			return None

		# Edit the pole polygon vertex
		# axle = self.axle
		
		# x = self.state
		# cartx = x[0] + screen_width / 2.0  # MIDDLE OF CART
		# self.carttrans.set_translation(cartx, carty)
		# self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def reset(self):
		'''
		This function resets the environment by setting the states, time_steps to zero
		'''
		self.state = np.zeros(2, dtype=np.int64)
		self.done = False
		self.time_steps = 0
		self.trajectory = [0]

		return self.state, self.time_steps