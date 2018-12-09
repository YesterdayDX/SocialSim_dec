import numpy as np
import random

class training_index:
	def __init__(self, n):
		self.n = n
		self.index = [i for i in range(self.n)]
		self.current_ind = 0

	def next(self, batch_size):
		if self.current_ind + batch_size < self.n:
			res_index = self.index[self.current_ind:self.current_ind+batch_size]
			self.current_ind += batch_size
		else:
			res_index = self.index[-batch_size:]
			self.current_ind = 0
			random.shuffle(self.index)
		return res_index

