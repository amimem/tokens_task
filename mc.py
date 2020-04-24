import gym
import numpy as np
import operator
import gym_tokens
import utils
from collections import defaultdict
import sys

# Source:
# https://github.com/dennybritz/reinforcement-learning/tree/master/MC
# On-Policy first-visit mc control with epsilon soft policies

def make_epsilon_greedy_policy(Q, epsilon, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function and epsilon.
	
	Args:
		Q: A dictionary that maps from state -> action-values.
			Each value is a numpy array of length nA (see below)
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.
	
	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length nA.
	
	"""
	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[tuple(observation)])
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.8, epsilon=0.01):
	"""
	Monte Carlo Control using Epsilon-Greedy policies.
	Finds an optimal epsilon-greedy policy.
	
	Args:
		env: OpenAI gym environment.
		num_episodes: Number of episodes to sample.
		discount_factor: Gamma discount factor.
		epsilon: Chance the sample a random action. Float betwen 0 and 1.
	
	Returns:
		A tuple (Q, policy).
		Q is a dictionary mapping state -> action values.
		policy is a function that takes an observation as an argument and returns
		action probabilities
	"""
	
	# Keeps track of sum and count of returns for each state
	# to calculate an average. We could use an array to save all
	# returns (like in the book) but that's memory inefficient.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)
	
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	
	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	rew = 0
	for i_episode in range(1, num_episodes + 1):
		# Print out which episode we're on, useful for debugging.
		if i_episode % 1000 == 0:
			print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
			sys.stdout.flush()

		# Generate an episode.
		# An episode is an array of (state, action, reward) tuples
		episode = []
		state, _ = env.reset()
		for t in range(100):
			probs = policy(state)
			action = np.random.choice(np.arange(len(probs)), p=probs)
			next_state, reward, done, _ = env.step(_mapFromIndexToTrueActions(action))
			episode.append((state, action, reward))
			if done:
				rew += reward
				break
			state = next_state
		
		print(rew/i_episode)

		# Find all (state, action) pairs we've visited in this episode
		# We convert each state to a tuple so that we can use it as a dict key
		sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
		for state, action in sa_in_episode:
			sa_pair = (state, action)
			# Find the first occurance of the (state, action) pair in the episode
			first_occurence_idx = next(i for i,x in enumerate(episode) if (x[0] == state).all() and x[1] == action)
			# Sum up all rewards since the first occurance
			G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
			# Calculate average return for this state over all sampled episodes
			returns_sum[sa_pair] += G
			returns_count[sa_pair] += 1.0
			Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
		
		# The policy is improved implicitly by changing the Q dictionary
	
	return Q, policy

def _mapFromIndexToTrueActions(actions):
	if actions == 1:
		return -1 
	elif actions == 2:
		return 1
	else:
		return 0

def _mapFromTrueActionsToIndex(actions):
	if actions == -1:
		return 1 
	elif actions == 1:
		return 2
	else:
		return 0 

if __name__ == "__main__":
	gamma = 0.8
	seed = 0

	utils.seed(seed)
	env = gym.make('tokens-v0', gamma=0.75, seed=seed, terminal=15, fancy_discount=False)
	env.seed(seed)
	Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.1)