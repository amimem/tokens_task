import numpy as np

class Policy:
	"""
	Abstract class that converts Q-values to actions
	"""

	def __call__(self, scores):
		raise NotImplementedError

class GreedyPolicy(Policy):
	"""
	Select actions that maximizes the Q-values
	"""

	def __call__(self, scores):
		assert isinstance(scores, np.ndarray)
		return np.argmax(scores, axis=0)

class EpsilonGreedyPolicy(Policy):
	"""
	Select random actions with prob <= epsilon, else select greedy actions
	"""

	def __init__(self, epsilon=0.01, default_policy=GreedyPolicy()):
		self.epsilon = epsilon
		self.default_policy = default_policy

	def __call__(self, scores):
		assert isinstance(scores, np.ndarray)

		num_actions = len(scores)
		eps_mask = np.random.random(size=1) < self.epsilon
		actions = self.default_policy(scores)
		rand_actions = np.random.choice(num_actions, size=sum(eps_mask))
		
		if eps_mask:

			return rand_actions

		else:

			return actions

		# return actions[0] #NEED TO FIX THIS TO USING NP ARRAY


class EpsilonTracker():

	def __init__(self, eps_start, eps_final, num_frames, policy):
		self.eps_start = eps_start
		self.eps_final = eps_final
		self.num_frames = num_frames
		self.policy = policy

	def set_eps(self, frame):
		eps = self.eps_start - frame/float(self.num_frames)
		self.policy.epsilon =  max(eps, self.eps_final)



