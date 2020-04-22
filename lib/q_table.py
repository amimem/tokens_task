import numpy as np

class Q_Table:

	def __init__(self, num_states, num_actions, shape, converge_val, height): 
		self.shape = shape
		# self.prev_q_matrix = np.zeros((num_states, num_actions))
		self.q_matrix = np.ones((num_states, num_actions))*0.5
		self.converge_val = converge_val
		self.height = height

	def get_qVal(self, states):
		statesID = self.get_stateID(states)
		return self.q_matrix[statesID, :]

	def update_qVal(self, learning_rate, states, actions, td_error):

		temp = self.q_matrix.copy()

		statesID = self.get_stateID(states)
		currentActionsIndex = self._mapFromTrueActionsToIndex(actions)
		self.q_matrix[statesID, currentActionsIndex] += learning_rate * td_error

		if self._hasConverged(temp, self.q_matrix):
			return True
		else:
			return False


	def _hasConverged(self, q_mat1, q_mat2):
		diff = np.linalg.norm(q_mat1 - q_mat2)

		if abs(diff) < self.converge_val:
			return True

		else: 
			return False


	def get_stateID(self,states):

		assert isinstance(states, np.ndarray)

		ids = []

		if len(self.shape == 3):
			num_rows, num_cols, timestep = self.shape

			Nt = self._augState(states[0])
			ht = self._augState(states[1])
			temp_id = Nt * num_cols + ht
			temp_id_time = temp_id*(timestep+1) + states[2]
		else:
			num_rows, num_cols = self.shape
			Nt = self._augState(states[0])
			ht = self._augState(states[1])
			temp_id = Nt * num_cols + ht
			temp_id_time = temp_id
			
		return temp_id_time

	def get_TDerror(self, states, actions, next_states, next_actions, reward, gamma, is_done, algo, model2 = None):
		statesID = self.get_stateID(states)
		currentActionsIndex = self._mapFromTrueActionsToIndex(actions)
		current_qVal = self.q_matrix[statesID, currentActionsIndex]

		if is_done:
			next_qVal = 0
		else:
			if algo == 'sarsa':
				next_statesID = self.get_stateID(next_states)
				nextActionsIndex = self._mapFromTrueActionsToIndex(next_actions)
				next_qVal = self.q_matrix[next_statesID, nextActionsIndex]

			elif algo == 'q-learning':
				next_statesID = self.get_stateID(next_states)
				next_qVal = np.max(self.q_matrix[next_statesID, :])

			elif algo == 'e-sarsa':
				next_statesID = self.get_stateID(next_states)
				next_qVal = np.sum(np.multiply(self.q_matrix[next_statesID, :], next_actions))

			elif algo == 'double-q':
				next_statesID = self.get_stateID(next_states)
				next_qVal = model2.q_matrix[next_statesID, np.argmax(self.q_matrix[next_statesID, :])]

		return reward + (gamma*next_qVal) - current_qVal

	def save_q_state(self, file, timestep):
		np.save(file+'/q_mat_'+str(timestep), self.q_matrix)

	def _augState(self, stateVal):
		"""
		Eg. Augment state value so that [-15,15] goes to [0,30] 
		"""
		return stateVal + self.height

	def _mapFromTrueActionsToIndex(self, actions):
		if actions == -1:
			return 1 
		elif actions == 1:
			return 2
		else:
			return 0 



