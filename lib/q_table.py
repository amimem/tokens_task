import numpy as np

class Q_Table:

	def __init__(self, num_states, num_actions, shape, converge_val): 
		self.shape = shape
		self.prev_q_matrix = np.zeros((num_states, num_actions))
		self.q_matrix = np.zeros((num_states, num_actions))
		self.converge_val = converge_val

	def get_qVal(self, states):
		statesID = self.get_stateID(states)
		return self.q_matrix[statesID, :]

	def update_qVal(self, learning_rate, states, actions, td_error):
		statesID = self.get_stateID(states)
		self.q_matrix[states,actions] = self.prev_q_matrix[states,actions] + learning_rate * (td_error)
		self.prev_q_matrix = self.q_matrix
		# if self.hasConverged(self.prev_q_matrix, self.q_matrix):
		# 	return True
		# else:
		# 	self.prev_q_matrix = self.q_matrix
		# 	return False


	def hasConverged(self, q_mat1, q_mat2):
		diff = np.linalg.norm(q_mat1 - q_mat2)
		
		if abs(diff) < self.converge_val:
			return True

		else: 
			return False


	def get_stateID(self,states):

		assert isinstance(states, list)

		ids = []

		num_rows, num_cols = self.shape		
		for state in states:
			temp_id = state[0] * num_cols + state[1]
			ids.append(temp_id)

		return ids

	def get_TDerror(self, states, actions, next_states, next_actions, reward):
		statesID = self.get_stateID(states)
		next_statesID = self.get_stateID(next_states)

		current_qVal = self.q_matrix[statesID, actions]
		next_qVal = self.q_matrix[next_statesID, next_actions]

		return reward + next_qVal - current_qVal

	def save_q_state(self, file):
		np.save(file+'/q_mat', self.q_matrix)



