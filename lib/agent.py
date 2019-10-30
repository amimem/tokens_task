"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np

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
	def __init__(self, policy_type, model):
		self.policy_type = policy_type
		self.model = model

	def get_actions(self, states):
		q_val = self.model.get_qVal(states)
		actions = self.policy_type(q_val)
		return self._mapFromIndexToTrueActions(actions)

	def _mapFromIndexToTrueActions(self, actions):
		if actions == 1:
			return -1 
		elif actions == 2:
			return 1
		else:
			return 0


	



		
