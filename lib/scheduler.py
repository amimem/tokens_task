class LRscheduler():

	def __init__(self, lr_start, lr_final, num_frames):
		self.lr_start = lr_start
		self.lr_final = lr_final
		self.num_frames = num_frames

	def get_lr(self, frame):
		lr = self.lr_start - frame/float(self.num_frames)
		return max(lr, self.lr_final)
