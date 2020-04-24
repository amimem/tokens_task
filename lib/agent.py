"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import random

class BaseAgent:
	"""
	Abstract Agent interface
	"""
	def initial_state(self):
		"""
		return initial state if any
		"""
		return None

	def __call__(self, states):
		"""
		Convert state into values
		"""
		assert isinstance(states, list)
		
		raise NotImplementedError

class SarsaAgent:
	"""
	Sarsa is an on-policy agent which updates the Q-values using latest experience
	"""
	def __init__(self, policy_type, model, max_steps):
		self.policy_type = policy_type
		self.model = model
		self.max_steps = max_steps

	def get_actions(self, states, get_probs=False, game_time_step=None):
		q_val = self.model.get_qVal(states)
		actions = self.policy_type(q_val)[0]
		action_mapped = self._mapFromIndexToTrueActions(actions)

		return action_mapped

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0

class QlAgent:
	"""
	Q-learning is an off-policy agent which updates the Q-values using max over all possible actions
	"""
	def __init__(self, policy_type, model, max_steps):
		self.policy_type = policy_type
		self.model = model
		self.max_steps = max_steps

	def get_actions(self, states, get_probs=False, game_time_step=None):
		q_val = self.model.get_qVal(states)
		actions = self.policy_type(q_val)[0]
		action_mapped = self._mapFromIndexToTrueActions(actions)
		return action_mapped

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0
class ExpectedSARSA:
	"""
	Expected SARSA
	"""
	def __init__(self, policy_type, model, max_steps):
		self.policy_type = policy_type
		self.model = model
		self.max_steps = max_steps

	def get_actions(self, states, get_probs=False, game_time_step=None):
		q_val = self.model.get_qVal(states)
		actions, probs = self.policy_type(q_val)
		action_mapped = self._mapFromIndexToTrueActions(actions)

		if get_probs:
			return action_mapped, probs
		else:
			return action_mapped

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0		

class DoubleQLearning:
	"""
	Double Q-Learning
	"""
	def __init__(self, policy_type, model1 , model2, max_steps):
		self.policy_type = policy_type
		self.model1 = model1
		self.model2 = model2
		self.max_steps = max_steps

	def get_actions(self, states, get_probs=False, game_time_step=None):
		q_val = self.model1.get_qVal(states) + self.model2.get_qVal(states)
		actions, probs = self.policy_type(q_val)
		action_mapped = self._mapFromIndexToTrueActions(actions)

		if get_probs:
			return action_mapped, probs
		else:
			return action_mapped

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0

class SemiSARSA:
	"""
	Semi-Gradient SARSA
	"""
	def __init__(self, policy_type, model, max_steps):
		self.policy_type = policy_type
		self.model = model
		self.max_steps = max_steps

	def get_actions(self, states, get_probs=False, game_time_step=None, shape=None):
		q_val = np.array([ self.model.get_qVal(self._one_hot(states,a,shape)) for a in range(shape[-1]) ])
		actions = self.policy_type(q_val)[0]
		action_mapped = self._mapFromIndexToTrueActions(actions)
		return action_mapped

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0

	def _mapFromTrueActionsToIndex(self, actions):
		if actions == -1:
			return 1 
		elif actions == 1:
			return 2
		else:
			return 0 

	def _augState(self, stateVal, height):
		"""
		Eg. Augment state value so that [-15,15] goes to [0,30] 
		"""
		return stateVal + height

	def _one_hot(self, state, action, shape):

		if len(shape) == 4:
			Nt, ht, z, A = shape

			x = np.zeros((Nt+ht+z)*A)

			one = np.eye(Nt)[self._augState(state[0],self.max_steps)]
			two = np.eye(ht)[self._augState(state[1],self.max_steps)]
			three = np.eye(z)[state[2]]

			a = self._mapFromTrueActionsToIndex(action)
			x[(Nt+ht+z)*a:(Nt+ht+z)*(a+1)] = np.hstack((one,two,three))

		else:
			Nt, ht, A = shape

			x = np.zeros((Nt+ht)*A)

			one = np.eye(Nt)[self._augState(state[0],self.max_steps)]
			two = np.eye(ht)[self._augState(state[1],self.max_steps)]

			a = self._mapFromTrueActionsToIndex(action)
			x[(Nt+ht)*a:(Nt+ht)*(a+1)] = np.hstack((one,two))

		return x